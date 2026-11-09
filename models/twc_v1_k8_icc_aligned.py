"""Independent K8 adaptation of the pinned ICC ordered-Swin/pilot-U-Net Joint.

Reference: 08 commit b8ea3a51b051b7aae22916d99aa52d811edb0850,
round6_resource/codec.py, round8/decoder.py and ul_unet/estimator.py.
Only public receiver inputs enter ``K8ICCAlignedJointDecoder``.  Simulation,
normalization, statistical recovery and successful-update EMA commits live in
the independent physical/runtime modules, never inside a network forward.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.baseline_common import angular_delay_transform, inverse_angular_delay_transform
from models.swin_precoder import PatchEmbedding, SwinTransformerBlock
from models.twc_v1_k8_joint_decoder import _ChunkedAttention, _coordinates


ARCHITECTURE = "twc_v1_k8_icc_aligned_v1"
K, NT, NSC, DMAX, WIDTH = 8, 32, 256, 216, 128
ARMS = ("fdma", "hybrid_g4")
REFERENCE_COMMIT = "b8ea3a51b051b7aae22916d99aa52d811edb0850"


def _protocol():
    # Avoid utils/__init__.py, which imports Sionna and performs GPU setup.
    name = "_twc_v1_k8_protocol"
    path = Path(__file__).resolve().parents[1] / "utils/twc_v1_k8_protocol.py"
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path:
            raise RuntimeError("K8 public protocol identity collision")
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def public_layout(arm):
    if arm not in ARMS:
        raise ValueError("Only fdma and hybrid_g4 are prescribed")
    p = _protocol()
    return p.FDMALayout() if arm == "fdma" else p.default_hybrid_layout(4)


def ordered_window_pool(tokens):
    """Pool actual 4x4 windows, retaining row-major 16x2 window order.

    Channel width may be tiny in component tests; production width is128.
    No pooling across windows and no global-mean replacement is performed.
    """
    if (not torch.is_tensor(tokens) or tokens.ndim != 3 or tokens.shape[0] < 1
            or tokens.shape[1] != 512 or tokens.shape[2] < 1
            or not tokens.is_floating_point() or not bool(torch.isfinite(tokens).all())):
        raise ValueError("Finite floating tokens[B,512,C] required")
    return tokens.reshape(tokens.shape[0], 16, 4, 2, 4, tokens.shape[2]).mean(
        dim=(2, 4)).reshape(tokens.shape[0], 32, tokens.shape[2])


def _gamma(gamma_db, batch, device):
    original = torch.as_tensor(gamma_db)
    if torch.is_complex(original) or original.dtype == torch.bool or original.requires_grad:
        raise ValueError("Public gamma must be real, nonlearned and nonboolean")
    value = torch.as_tensor(gamma_db, dtype=torch.float32, device=device)
    if value.ndim == 0:
        value = value.expand(batch)
    if value.shape != (batch,) or not bool(torch.isfinite(value).all() & (value >= 0).all() & (value <= 25).all()):
        raise ValueError("Public gamma must be scalar or [B] in [0,25]dB")
    return value


def _complex(value, shape, name, device=None):
    if (not shape or shape[0] < 1 or not torch.is_tensor(value) or tuple(value.shape) != tuple(shape)
            or value.dtype != torch.complex64 or not bool(torch.isfinite(value).all())
            or device is not None and value.device != device):
        raise ValueError("Finite complex64 public " + name + " has wrong shape/device")


class K8ICCAlignedEncoder(nn.Module):
    """Full32x256 CSI -> four Swin -> ordered4096 -> common216 complex slots."""
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(2, WIDTH, 4)
        self.blocks = nn.ModuleList([SwinTransformerBlock(WIDTH, 4, 4, 4.) for _ in range(4)])
        self.norm = nn.LayerNorm(WIDTH)
        self.fc = nn.Linear(4096, 2 * DMAX)

    def forward(self, H_dl):
        batch = H_dl.shape[0] if torch.is_tensor(H_dl) and H_dl.ndim else 0
        _complex(H_dl, (batch, K, NT, NSC), "UE source")
        features = angular_delay_transform(H_dl, NSC).reshape(batch*K, 2, NSC, NT)
        tokens, height, width = self.patch_embed(features)
        if (height, width) != (64, 8):
            raise ValueError("Full-source token grid must be64x8")
        for block in self.blocks:
            tokens = block(tokens, height, width)
        output = self.fc(ordered_window_pool(self.norm(tokens)).flatten(1)).reshape(batch, K, 2*DMAX)
        return torch.complex(output[..., :DMAX], output[..., DMAX:])


class _CEBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(inputs, outputs, 5, padding=2), nn.GroupNorm(8, outputs), nn.GELU(),
            nn.Conv1d(outputs, outputs, 5, padding=2), nn.GroupNorm(8, outputs), nn.GELU())

    def forward(self, value):
        return self.layers(value)


class K8ICCAlignedPilotUNet(nn.Module):
    """New133-channel ICC pilot-only residual CE, with K8 and gamma0..25.

    Physical pilots have amplitude sqrt(16/8); features include these known
    amplitudes.  Y is masked before checks/features, so data-only values,
    including nonfinite values, cannot influence this network. Initial CE must
    be computed by the separate pilot-only Fourier rank8 estimator.
    """
    def __init__(self):
        super().__init__()
        mask = torch.zeros(K, NSC, dtype=torch.bool)
        for user, positions in enumerate(_protocol().PILOT_POSITIONS_BY_USER):
            mask[user, list(positions)] = True
        self.register_buffer("pilot_mask", mask, persistent=False)
        self.register_buffer("pilot_union", mask.any(0), persistent=False)
        self.encoder0 = _CEBlock(133, 32)
        self.down1 = nn.Conv1d(32, 64, 5, stride=2, padding=2)
        self.encoder1 = _CEBlock(64, 64)
        self.down2 = nn.Conv1d(64, 96, 5, stride=2, padding=2)
        self.bottleneck = _CEBlock(96, 96)
        self.decoder1 = _CEBlock(96+64, 64)
        self.decoder0 = _CEBlock(64+32, 32)
        self.head = nn.Conv1d(32, 2*NT, 1)
        nn.init.normal_(self.head.weight, mean=0., std=1e-4)
        nn.init.zeros_(self.head.bias)

    def features(self, Y, pilots, H_initial, *, gamma_db):
        batch = Y.shape[0] if torch.is_tensor(Y) and Y.ndim else 0
        if not torch.is_tensor(Y) or Y.dtype != torch.complex64 or tuple(Y.shape) != (batch, NT, NSC):
            raise ValueError("CE requires measured complex64Y[B,32,256]")
        _complex(pilots, (batch, K, NSC), "known pilots", Y.device)
        _complex(H_initial, (batch, K, NT, NSC), "pilot-only initial CE", Y.device)
        if Y.device != self.head.weight.device or self.head.weight.dtype != torch.float32:
            raise ValueError("CE and public inputs must share an FP32 device")
        if pilots.requires_grad:
            raise ValueError("Known pilot amplitudes cannot be trainable")
        own, union = self.pilot_mask, self.pilot_union
        pilot_Y = torch.where(union[None, None], Y, torch.zeros((), dtype=Y.dtype, device=Y.device))
        amplitudes = pilots[:, own].abs()
        if (not bool(torch.isfinite(pilot_Y).all()) or bool((pilots[:, ~own] != 0).any())
                or not bool(torch.isclose(amplitudes, amplitudes.new_full(amplitudes.shape, 2.**.5), rtol=1e-6, atol=1e-6).all())):
            raise ValueError("Only finite pilot-supported observations and Ep16 known pilots are allowed")
        gamma = _gamma(gamma_db, batch, Y.device)
        # K8 adaptation centers the explicitly extended0..25 range at12.5.
        snr = (gamma - 12.5) / 12.5
        common = torch.cat((pilot_Y.real, pilot_Y.imag), 1)[:, None].expand(-1, K, -1, -1)
        initial = torch.cat((H_initial.real, H_initial.imag), 2)
        return torch.cat((common, initial, pilots.real[:, :, None], pilots.imag[:, :, None],
            own[None, :, None].expand(batch, -1, -1, -1),
            union[None, None, None].expand(batch, K, -1, -1),
            snr[:, None, None, None].expand(batch, K, 1, NSC)), 2).reshape(batch*K, 133, NSC)

    _features = features

    def forward(self, Y, pilots, H_initial, *, gamma_db):
        inputs = self.features(Y, pilots, H_initial, gamma_db=gamma_db)
        x0 = self.encoder0(inputs)
        x1 = self.encoder1(self.down1(x0))
        bottom = self.bottleneck(self.down2(x1))
        up1 = self.decoder1(torch.cat((F.interpolate(bottom, size=x1.shape[-1], mode="linear", align_corners=False), x1), 1))
        up0 = self.decoder0(torch.cat((F.interpolate(up1, size=x0.shape[-1], mode="linear", align_corners=False), x0), 1))
        residual = self.head(up0).reshape(Y.shape[0], K, 2*NT, NSC)
        corrected = H_initial + torch.complex(residual[:, :, :NT], residual[:, :, NT:])
        if not bool(torch.isfinite(corrected).all()):
            raise FloatingPointError("Nonfinite full-frequency CE correction")
        return corrected

    def metadata(self):
        return dict(architecture="icc_pilot_residual_unet_K8_v1", width=[32,64,96], input_channels=133,
            groupnorm=8, head_initial_std=1e-4, pilot_only=True, pilot_RE=64,
            gamma_feature="(gamma_db-12.5)/12.5", output_projected_back_to_rank8=False,
            trainable_parameters=sum(p.numel() for p in self.parameters()))


def support_geometry(layout):
    """Return fixed216-slot public maps; never inspect simulator values."""
    support = torch.full((K, DMAX), -1, dtype=torch.long)
    exclusive = torch.zeros_like(support, dtype=torch.bool)
    if hasattr(layout, "data_positions"):
        for user, positions in enumerate(layout.data_positions):
            support[user, :24] = torch.as_tensor(positions)
            exclusive[user, :24] = True
    elif hasattr(layout, "shared_positions_by_group"):
        for user, positions in enumerate(layout.key_positions):
            support[user, :4] = torch.as_tensor(positions)
            exclusive[user, :4] = True
        for members, positions in zip(layout.groups, layout.shared_positions_by_group):
            for user in members:
                support[user, 4:44] = torch.as_tensor(positions)
    else:
        raise TypeError("Only fixed K8 FDMA or G4 public layouts are accepted")
    active = support >= 0
    return support, active, exclusive


class _ICCJointLayer(nn.Module):
    """ICC pre-norm self-attention, observation attention, GELU-FFN."""
    def __init__(self, width=WIDTH, heads=4, query_chunk=64, activation_checkpointing=True):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = _ChunkedAttention(width=width, heads=heads, query_chunk=query_chunk,
            activation_checkpoint=activation_checkpointing)
        self.cross_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = _ChunkedAttention(width=width, heads=heads, query_chunk=query_chunk,
            activation_checkpoint=activation_checkpointing)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))

    def forward(self, query, memory, active, self_bias=None, cross_bias=None):
        normalized = self.self_norm(query)
        query = (query + self.self_attention(normalized, normalized, bias_builder=self_bias, key_mask=active))*active[..., None]
        query = (query + self.cross_attention(self.cross_norm(query), self.memory_norm(memory), bias_builder=cross_bias))*active[..., None]
        return (query + self.mlp(self.mlp_norm(query)))*active[..., None]


class K8ICCAlignedJointDecoder(nn.Module):
    """Two all-user Joint layers -> latent residual -> FC/spatial cross ->4Swin.

    No current-frame tx energy, true UL/DL, noiseless Y or true symbols are
    accepted. ``prior``/``ce_error`` are optional public historical diagnostics;
    their receiver effect is already present in analytical soft/reliability.
    """
    def __init__(self, arm, *, activation_checkpointing=True):
        super().__init__()
        self.arm = arm
        self.activation_checkpointing = bool(activation_checkpointing)
        self.fc = nn.Linear(2*DMAX, 512*WIDTH)
        self.blocks = nn.ModuleList([SwinTransformerBlock(WIDTH, 4, 4, 4.) for _ in range(4)])
        self.norm = nn.LayerNorm(WIDTH)
        self.final_proj = nn.Linear(WIDTH, 32)
        self.memory_y = nn.Linear(2*NT, WIDTH)
        self.memory_channel = nn.Linear(2*K*NT, WIDTH)
        self.memory_pilot = nn.Linear(2*K, WIDTH)
        self.memory_position = nn.Linear(3, WIDTH)
        self.memory_reliability = nn.Linear(K, WIDTH)
        self.memory_noise = nn.Linear(1, WIDTH)
        self.memory_norm = nn.LayerNorm(WIDTH)
        self.query_projection = nn.Linear(K+11, WIDTH)
        self.exclusive_summary = nn.Linear(4, WIDTH)
        self.joint_layers = nn.ModuleList([_ICCJointLayer(activation_checkpointing=activation_checkpointing) for _ in range(2)])
        self.overlap_bias = nn.Parameter(torch.full((4,), .02))
        self.coupling_bias = nn.Parameter(torch.full((4,), .02))
        self.resource_cross_bias = nn.Parameter(torch.full((4,), .02))
        self.residual_norm = nn.LayerNorm(WIDTH)
        self.residual_head = nn.Linear(WIDTH, 2)
        self.spatial_position = nn.Linear(K+6, WIDTH)
        self.spatial_query_norm = nn.LayerNorm(WIDTH)
        self.spatial_context_norm = nn.LayerNorm(WIDTH)
        self.spatial_attention = _ChunkedAttention(query_chunk=64, activation_checkpoint=activation_checkpointing)
        # As in ICC, do not reset the source spatial FC/Swin/readout weights.
        source = {"fc", "blocks", "norm", "final_proj"}
        for name, module in self.named_modules():
            if name.split(".")[0] in source:
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.residual_head.weight, std=.001)
        nn.init.normal_(self.spatial_attention.out.weight, std=.001)
        support, active, exclusive = support_geometry(public_layout(arm))
        for name, value in dict(support=support, active=active, exclusive=exclusive).items():
            self.register_buffer("_fixed_"+name, value, persistent=False)

    def forward(self, Y_rx, H_ul_est, soft, reliability, pilots, *, gamma_db, layout=None, prior=None, ce_error=None):
        batch = Y_rx.shape[0] if torch.is_tensor(Y_rx) and Y_rx.ndim else 0
        device = self.fc.weight.device
        for name, value, shape in (("Y",Y_rx,(batch,NT,NSC)), ("estimated UL",H_ul_est,(batch,K,NT,NSC)),
                ("soft",soft,(batch,K,DMAX)), ("pilots",pilots,(batch,K,NSC))):
            _complex(value, shape, name, device)
        if (not torch.is_tensor(reliability) or reliability.shape != (batch,K,DMAX)
                or reliability.dtype != torch.float32 or reliability.device != device or not bool(torch.isfinite(reliability).all())
                or bool((reliability[:, self._fixed_active] <= 0).any()) or bool((reliability[:, ~self._fixed_active] != 0).any())):
            raise ValueError("Historical calibrated receiver precision must be positive on active slots and zero in padding")
        if layout is not None and any(not torch.equal(g.to(device), getattr(self,"_fixed_"+n))
                for g,n in zip(support_geometry(layout), ("support","active","exclusive"))):
            raise ValueError("Public layout differs from this fixed arm")
        gamma = _gamma(gamma_db, batch, device)
        active = self._fixed_active
        support = self._fixed_support.clamp_min(0)
        soft = soft*active[None]
        precision = torch.log1p(reliability)*active[None]
        onehot = F.one_hot(support, NSC).to(torch.float32)*active[...,None]
        grid = torch.einsum("bkd,kdl->blk",precision,onehot)
        y = torch.view_as_real(Y_rx.transpose(1,2).contiguous()).flatten(-2)
        h = torch.view_as_real(H_ul_est.permute(0,3,1,2).contiguous()).flatten(-3)
        pilot = torch.view_as_real(pilots.transpose(1,2).contiguous()).flatten(-2)
        re = torch.arange(NSC,device=device)
        memory = self.memory_norm(self.memory_y(y)+self.memory_channel(h)+self.memory_pilot(pilot)
            +self.memory_position(_coordinates(re,NSC))[None]+self.memory_reliability(grid)
            +self.memory_noise(((gamma-12.5)/12.5)[:,None,None].expand(-1,NSC,1)))
        identity = torch.eye(K,device=device)[:,None].expand(-1,DMAX,-1)
        slots = _coordinates(torch.arange(DMAX,device=device),DMAX)[None].expand(K,-1,-1)
        roles = torch.stack((self._fixed_exclusive, active & ~self._fixed_exclusive),-1).float()
        fixed = torch.cat((_coordinates(support,NSC),slots,roles,identity),-1)[None].expand(batch,-1,-1,-1)
        query = self.query_projection(torch.cat((torch.view_as_real(soft),fixed,precision[...,None]),-1))
        exclusive = self._fixed_exclusive[None]
        count = exclusive.sum(-1).clamp_min(1)
        mean = (soft*exclusive).sum(-1)/count
        summary = torch.stack((mean.real,mean.imag,(soft.abs().square()*exclusive).sum(-1)/count,
            (soft.abs()*exclusive).sum(-1)/count),-1)
        query = ((query+self.exclusive_summary(summary)[:,:,None])*active[None,:,:,None]).flatten(1,2)
        channel = H_ul_est.permute(0,3,1,2)
        gram = channel@channel.conj().transpose(-1,-2)
        norm = channel.abs().square().sum(-1).clamp_min(1e-12)
        coupling = (gram.abs().square()/(norm[...,:,None]*norm[...,None,:])).mean(1)
        users = torch.arange(K,device=device).repeat_interleave(DMAX)
        resources = support.flatten()
        def self_bias(start,end):
            overlap = (resources[start:end,None] == resources[None]).float()
            coupled = coupling[:,users[start:end,None],users[None]]
            return (overlap[None,None]*self.overlap_bias[None,:,None,None]
                +coupled[:,None]*self.coupling_bias[None,:,None,None])
        def cross_bias(start,end):
            return (resources[start:end,None] == re[None]).float()[None,None]*self.resource_cross_bias[None,:,None,None]
        active_flat = active.flatten()[None].expand(batch,-1)
        for layer in self.joint_layers:
            # Layer internals use nonreentrant checkpoints at query chunks;
            # all keys remain visible, never local or per-user attention.
            query = layer(query,memory,active_flat,self_bias,cross_bias)
        residual = self.residual_head(self.residual_norm(query)).reshape(batch,K,DMAX,2)
        latent = (soft+torch.view_as_complex(residual.contiguous()))*active[None]
        codeword = torch.cat((latent.real,latent.imag),-1).reshape(batch*K,2*DMAX)
        spatial = self.fc(codeword).reshape(batch,K*512,WIDTH)
        rows = _coordinates(torch.arange(64,device=device),64)[:,None].expand(64,8,3)
        columns = _coordinates(torch.arange(8,device=device),8)[None].expand(64,8,3)
        coords = torch.cat((rows,columns),-1).reshape(1,512,6).expand(K,-1,-1)
        uid = torch.eye(K,device=device)[:,None].expand(K,512,K)
        spatial = spatial+self.spatial_position(torch.cat((coords,uid),-1)).reshape(1,K*512,WIDTH)
        spatial = spatial+self.spatial_attention(self.spatial_query_norm(spatial),self.spatial_context_norm(query),key_mask=active_flat)
        tokens = spatial.reshape(batch*K,512,WIDTH)
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block,tokens,64,8,use_reentrant=False)
            else:
                tokens = block(tokens,64,8)
        patches = self.final_proj(self.norm(tokens)).reshape(batch*K,64,8,2,4,4)
        features = patches.permute(0,3,1,4,2,5).contiguous().reshape(batch,K,2,NSC,NT)
        H_hat = inverse_angular_delay_transform(features,NSC)
        if not bool(torch.isfinite(H_hat).all() & torch.isfinite(latent).all()):
            raise FloatingPointError("Nonfinite full physical DL CSI/reconstruction latent")
        return dict(H_hat=H_hat,reconstruction_latent=latent,diagnostics=dict(
            observation_memory_shape=list(memory.shape),joint_context_shape=list(query.shape),
            spatial_shape=[batch,K*512,WIDTH],joint_layers_executed=2,swin_blocks_executed=4,
            spatial_cross_attention_executed=True,raw_received_Y_used=True,
            same_corrected_UL_used=True,complete_delay_taps=NSC,
            latent_is_physical_tx=False,latent_renormalized=False,true_inputs_used=False))

    def metadata(self):
        return dict(architecture=ARCHITECTURE,arm=self.arm,Dmax=DMAX,width=WIDTH,joint_layers=2,
            heads=4,FFN=256,swin_blocks=4,patch=4,window=4,full_output=[K,NT,NSC],
            fc_input=432,fc_output=65536,query_slots=K*DMAX,spatial_queries=K*512,
            parameters=sum(p.numel() for p in self.parameters()),
            reference_commit=REFERENCE_COMMIT,unpaid_frame_energy_side_information=False)


class K8ICCAlignedCodec(nn.Module):
    def __init__(self, arm, *, activation_checkpointing=True):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Only the prescribed two new arms are supported")
        self.arm = arm
        self.encoder = K8ICCAlignedEncoder()
        self.decoder = K8ICCAlignedJointDecoder(arm,activation_checkpointing=activation_checkpointing)

    def encode(self,H_dl):
        return self.encoder(H_dl)

    def decode(self,*args,**kwargs):
        return self.decoder(*args,**kwargs)

    def metadata(self):
        return dict(architecture=ARCHITECTURE,arm=self.arm,reference_commit=REFERENCE_COMMIT,
            encoder_parameters=sum(p.numel() for p in self.encoder.parameters()),
            decoder=self.decoder.metadata(),total_parameters=sum(p.numel() for p in self.parameters()),
            encoder_ordered_windows=32,source_truncated=False,common_complex_interface=DMAX)


def create_codec(arm, *, device="cuda", seed=849301, activation_checkpointing=True):
    """Same seeded common weights, independent storage; no historical weights.

    Construction occurs once on CPU under isolated default CPU RNG, then moves
    this instance to its device. Two arms can be trained sequentially without
    holding duplicate full GPU models or changing their common initialization.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32-1:
        raise ValueError("Explicit uint32 common initialization seed required")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = K8ICCAlignedCodec(arm,activation_checkpointing=activation_checkpointing)
    return model.to(device)
