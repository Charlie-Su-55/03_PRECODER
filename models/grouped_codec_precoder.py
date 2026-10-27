"""Independent representation x grouped-layout precoder with an absolute-noise protocol.

No historical checkpoint or physical-noise convention is imported. Existing UE
layers/Swin blocks and normalized RZF are reused as structure only. The BS sees
only Y, perfect public UL CSI, mapped soft symbols/reliability and public layout.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from configs.grouped_codec_train_config import protocol
from .adaptive_hybrid import AdaptiveHybridEncoder, AdaptiveHybridPrecoder, ConditionEmbed
from .adaptive_hybrid_joint_swin import ChunkedAttention, _coordinates, _initialize_extra, _run, channel_coupling
from .baseline_common import angular_delay_transform, inverse_angular_delay_transform
from .swin_precoder import PatchEmbedding, SwinCFNetDecoder, SwinTransformerBlock


def ordered_window_pool(tokens):
    """[BK,64*8,128] -> [BK,16*2,128], row-major delay-window then angle-window."""
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (512, 128) or tokens.shape[0] < 1:
        raise ValueError("Ordered pooling requires nonempty [BK,512,128].")
    return tokens.reshape(tokens.shape[0], 16, 4, 2, 4, 128).mean((2, 4)).reshape(
        tokens.shape[0], 32, 128)


class LegacyUE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = AdaptiveHybridEncoder(cfg)
        self.condition = ConditionEmbed(256, 64, 256)

    def forward(self, H_dl, gamma_db, active_per_user):
        batch = H_dl.shape[0]
        # This is original UE conditioning, not a fictitious physical K*Df+Da
        # allocation. The full head is retained; selection is done by LayoutSpec.
        cond = self.condition(0, active_per_user, gamma_db, H_dl.device)
        _, full = self.encoder(H_dl.reshape(batch * 4, 32, 256), 0, 64, cond)
        return full.reshape(batch, 4, 256)


class OrderedWindowUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(2, 128, 4)
        self.blocks = nn.ModuleList([SwinTransformerBlock(128, 4, 4, 4.0) for _ in range(4)])
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Linear(4096, 512)

    def forward(self, H_dl, gamma_db, active_per_user):
        batch = H_dl.shape[0]
        full = angular_delay_transform(H_dl, 256).reshape(batch * 4, 2, 256, 32)
        tokens, height, width = self.patch_embed(full)
        if (height, width) != (64, 8):
            raise ValueError("Ordered UE must retain the full 64x8 token grid.")
        for block in self.blocks:
            tokens = _run(block, tokens, height, width, enabled=self.training)
        real = self.fc(ordered_window_pool(self.norm(tokens)).flatten(1))
        return torch.complex(real[:, :256], real[:, 256:]).reshape(batch, 4, 256)


class MappedAttention(ChunkedAttention):
    """All-key attention whose biases use explicit physical q, never slot%256."""

    def _mapped_chunk(self, query, key, value, support, active, coupling,
                      overlap_weight, coupling_weight, resource_weight, start, kind):
        q = self._split(self.q(query))
        bias = None
        if kind in ("self", "observation"):
            query_indices = torch.arange(start, start + query.shape[1], device=query.device)
            query_q = support[query_indices]
            key_q = support if kind == "self" else torch.arange(256, device=query.device)
            same = (query_q[:, None] == key_q[None, :]).to(query.dtype)
            if kind == "self":
                qi = query_indices.div(256, rounding_mode="floor")
                ki = torch.arange(key.shape[-2], device=query.device).div(256, rounding_mode="floor")
                overlaps = coupling[:, qi][:, :, ki]
                bias = (same[None, None] * overlap_weight[None, :, None, None]
                        + overlaps[:, None] * coupling_weight[None, :, None, None])
            else:
                bias = same[None, None] * resource_weight[None, :, None, None]
        if kind in ("self", "spatial"):
            if bias is None:
                bias = query.new_zeros((1, 1, 1, key.shape[-2]))
            bias = bias.masked_fill(~active[None, None, None, :], -torch.inf)
        if kind not in ("self", "observation", "spatial"):
            raise ValueError("Unknown mapped attention kind.")
        output = F.scaled_dot_product_attention(q, key, value, attn_mask=bias,
                                               dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.width)
        return self.out(output)

    def forward(self, query, memory, support, active, coupling=None, overlap_weight=None,
                coupling_weight=None, resource_weight=None, kind="spatial"):
        key, value = self._split(self.k(memory)), self._split(self.v(memory))
        chunks = []
        for start in range(0, query.shape[1], self.query_chunk):
            chunks.append(_run(
                self._mapped_chunk, query[:, start:start + self.query_chunk], key, value,
                support, active, coupling, overlap_weight, coupling_weight,
                resource_weight, start, kind,
                enabled=self.activation_checkpoint and self.training,
            ))
        return torch.cat(chunks, dim=1)


class MappedJointLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_norm = nn.LayerNorm(128)
        self.self_attention = MappedAttention()
        self.cross_norm = nn.LayerNorm(128)
        self.memory_norm = nn.LayerNorm(128)
        self.cross_attention = MappedAttention()
        self.mlp_norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, x, memory, support, active, coupling,
                overlap_weight, coupling_weight, resource_weight):
        mask = active[None, :, None]
        normalized = self.self_norm(x)
        x = (x + self.self_attention(
            normalized, normalized, support, active, coupling, overlap_weight,
            coupling_weight, resource_weight, "self"
        )) * mask
        x = (x + self.cross_attention(
            self.cross_norm(x), self.memory_norm(memory), support, active, coupling,
            overlap_weight, coupling_weight, resource_weight, "observation"
        )) * mask
        return (x + self.mlp(self.mlp_norm(x))) * mask


class GroupedObservationMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.y = nn.Linear(64, 128)
        self.channel = nn.Linear(256, 128)
        self.position = nn.Linear(3, 128)
        self.reliability = nn.Linear(4, 128)
        self.activity = nn.Linear(4, 128)
        self.snr = nn.Linear(1, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, Y, H_ul_est, reliability_grid, physical_mask, coordinates, gamma):
        batch = Y.shape[0]
        received = torch.view_as_real(Y.transpose(1, 2).contiguous()).flatten(-2)
        channel = torch.view_as_real(H_ul_est.permute(0, 3, 1, 2).contiguous()).flatten(-3)
        return self.norm(
            self.y(received) + self.channel(channel) + self.position(coordinates)[None]
            + self.reliability(reliability_grid)
            + self.activity(physical_mask.T.to(received.dtype))[None]
            + self.snr(gamma.expand(batch, 256, 1))
        )


class GroupedJointSwinDecoder(nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.codec = SwinCFNetDecoder(SimpleNamespace(
            SWIN_NC=256, ANTENNAS=32, SWIN_PATCH_SIZE=4, SWIN_WINDOW_SIZE=4,
            SWIN_EMBED_DIM=128, SWIN_NUM_HEADS=4, SWIN_MLP_RATIO=4.0,
            SWIN_NUM_BLOCKS=4, SWIN_CODEWORD_DIM=512,
        ))
        self.spatial_position = nn.Linear(10, 128)
        self.observation_memory = GroupedObservationMemory()
        self.query_projection = nn.Linear(15, 128)
        self.joint_layers = nn.ModuleList([MappedJointLayer() for _ in range(2)])
        self.overlap_bias = nn.Parameter(torch.full((4,), 0.02))
        self.coupling_bias = nn.Parameter(torch.full((4,), 0.02))
        self.resource_cross_bias = nn.Parameter(torch.full((4,), 0.02))
        self.residual_norm = nn.LayerNorm(128)
        self.residual_head = nn.Linear(128, 2)
        self.spatial_query_norm = nn.LayerNorm(128)
        self.spatial_context_norm = nn.LayerNorm(128)
        self.spatial_attention = MappedAttention()
        for name, module in self.named_modules():
            if name == "codec" or name.startswith("codec."):
                continue
            _initialize_extra(module)
        nn.init.normal_(self.residual_head.weight, std=0.001)
        nn.init.normal_(self.spatial_attention.out.weight, std=0.001)
        # Geometry creates no RNG draws and is the only layout-varying state.
        for name, value in protocol.tensor_geometry(layout).items():
            self.register_buffer(name, value, persistent=False)
        resource_coordinates = _coordinates(torch.arange(256), 256)
        one_hot = torch.eye(4)
        slots = resource_coordinates[None].expand(4, 256, 3)
        physical_coordinates = _coordinates(self.support_sc.clamp_min(0), 256)
        exclusive = self.active_mask if layout.layout == "fdma" else torch.zeros_like(self.active_mask)
        roles = torch.stack((exclusive, self.active_mask & ~exclusive), dim=-1).float()
        query_fixed = torch.cat((physical_coordinates, slots, roles,
                                 one_hot[:, None].expand(4, 256, 4)), dim=-1)
        rows, columns = _coordinates(torch.arange(64), 64), _coordinates(torch.arange(8), 8)
        spatial = torch.cat((rows[None, :, None].expand(4, 64, 8, 3),
                             columns[None, None].expand(4, 64, 8, 3),
                             one_hot[:, None, None].expand(4, 64, 8, 4)), dim=-1)
        self.register_buffer("resource_coordinates", resource_coordinates, persistent=False)
        self.register_buffer("query_fixed", query_fixed, persistent=False)
        self.register_buffer("spatial_coordinates", spatial.reshape(4, 512, 10), persistent=False)

    def _spatial_cross(self, spatial, context):
        return self.spatial_attention(
            self.spatial_query_norm(spatial), self.spatial_context_norm(context),
            self.support_sc.flatten(), self.active_mask.flatten(), kind="spatial"
        )

    def forward(self, z_hat, reliability, Y, H_ul_est, gamma_db):
        batch = Y.shape[0]
        if (tuple(Y.shape) != (batch, 32, 256)
                or tuple(H_ul_est.shape) != (batch, 4, 32, 256)
                or tuple(z_hat.shape) != (batch, 4, 256)
                or reliability.shape != z_hat.shape):
            raise ValueError("Grouped public decoder input shape mismatch.")
        device = self.codec.fc.weight.device
        for name, value in (("z_hat", z_hat), ("Y", Y), ("H_ul_est", H_ul_est)):
            if value.dtype != torch.complex64 or value.device != device:
                raise TypeError(name + " must be complex64 on the decoder device.")
        if reliability.dtype != torch.float32 or reliability.device != device:
            raise TypeError("Reliability must be float32 on the decoder device.")
        protocol.noise_variance(gamma_db)
        if not bool(((reliability >= 0) & (reliability <= 1)).all()):
            raise ValueError("Reliability proxy must lie in [0,1].")
        if any(not bool(torch.isfinite(value).all()) for value in (z_hat, reliability, Y, H_ul_est)):
            raise FloatingPointError("Nonfinite receiver input; no truth substitution.")
        mask = self.active_mask[None]
        soft = z_hat.masked_fill(~mask, 0)
        proxy = torch.log1p(reliability).masked_fill(~mask, 0)
        active_proxy = proxy.index_select(-1, self.active_slots)
        grid = proxy.new_zeros((batch, 4, 256)).scatter(
            -1, self.physical_supports[None].expand(batch, -1, -1), active_proxy).transpose(1, 2)
        gamma = proxy.new_full((1, 1, 1), float(gamma_db) / 25.0)
        memory = _run(self.observation_memory, Y, H_ul_est, grid, self.physical_mask,
                      self.resource_coordinates, gamma, enabled=self.training)
        features = torch.cat((torch.view_as_real(soft.contiguous()),
                              self.query_fixed[None].expand(batch, -1, -1, -1),
                              proxy[..., None]), dim=-1)
        context = (self.query_projection(features) * mask[..., None]).reshape(batch, 1024, 128)
        coupling = channel_coupling(H_ul_est)
        for layer in self.joint_layers:
            context = _run(layer, context, memory, self.support_sc.flatten(),
                           self.active_mask.flatten(), coupling, self.overlap_bias,
                           self.coupling_bias, self.resource_cross_bias, enabled=self.training)
        residual = self.residual_head(self.residual_norm(context)).reshape(batch, 4, 256, 2)
        correction = torch.complex(residual[..., 0], residual[..., 1]).masked_fill(~mask, 0)
        corrected = (soft + correction).masked_fill(~mask, 0)
        real = torch.cat((corrected.real, corrected.imag), dim=-1)
        spatial = self.codec.fc(real.reshape(batch * 4, 512)).reshape(batch, 4, 512, 128)
        spatial = spatial + self.spatial_position(self.spatial_coordinates)[None]
        flat = spatial.reshape(batch, 2048, 128)
        spatial = flat + _run(self._spatial_cross, flat, context, enabled=self.training)
        tokens = spatial.reshape(batch * 4, 512, 128)
        for block in self.codec.blocks:
            tokens = _run(block, tokens, 64, 8, enabled=self.training)
        patches = self.codec.final_proj(self.codec.norm(tokens)).reshape(batch * 4, 64, 8, 2, 4, 4)
        H_ad = patches.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(batch, 4, 2, 256, 32)
        if not bool(torch.isfinite(H_ad).all() & torch.isfinite(corrected).all()):
            raise FloatingPointError("Nonfinite complete Joint decoder output.")
        execution = {"memory_shape": list(memory.shape), "context_shape": list(context.shape),
                     "spatial_shape": list(spatial.shape), "angular_delay_shape": list(H_ad.shape)}
        return H_ad, corrected, execution


class GroupedCodecPrecoder(nn.Module):
    zf_solver = AdaptiveHybridPrecoder.zf_solver

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.arm = cfg.arm
        self.layout = protocol.layout_for_arm(self.arm)
        self.ue_frozen = False
        expected = {"K_USERS": 4, "ANTENNAS": 32, "SUBCARRIERS": 256,
                    "NUM_SUBBANDS": 4, "D_MODEL": 256, "NUM_HEADS": 8,
                    "NUM_ENCODER_LAYERS": 4, "DROPOUT": 0.1,
                    "D_F_MAX_PER_USER": 64, "D_A_MAX_SHARED": 256, "TOTAL_POWER": 1.0}
        for name, value in expected.items():
            if getattr(cfg, name, None) != value:
                raise ValueError("Grouped codec fixes " + name + "=" + str(value))
        family = "legacy" if self.arm.startswith("legacy_") else "ordered"
        # Both contexts preserve the caller's CPU RNG and never reseed CUDA.
        # Independent BS stream is identical for all five arms at each seed.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                protocol.RandomAddress(cfg.SEED, "init", 0, "ue_" + family).seed_value)
            self.ue = LegacyUE(cfg) if family == "legacy" else OrderedWindowUE()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                protocol.RandomAddress(cfg.SEED, "init", 0, "bs").seed_value)
            self.decoder = GroupedJointSwinDecoder(self.layout)

    @property
    def architecture_metadata(self):
        return protocol.architecture_metadata(self.arm)

    def ue_parameters(self):
        yield from self.ue.parameters()

    def bs_parameters(self):
        yield from self.decoder.parameters()

    def inactive_parameter_names(self):
        return tuple(name for name, _ in self.named_parameters()
                     if name.startswith(("ue.encoder.fdma_", "ue.encoder.fc_fdma.")))

    def _forward(self, H_dl, H_ul, gamma_db, *, standard_noise, H_ul_est=None, diagnostics=False):
        N0 = protocol.noise_variance(gamma_db)
        if H_dl.ndim != 4 or H_dl.shape[0] < 1 or tuple(H_dl.shape[1:]) != (4, 32, 256):
            raise ValueError("Expected nonempty H_dl[B,4,32,256].")
        protocol._complex_tensor(H_dl, "DL input")
        if H_ul.shape != H_dl.shape or H_ul.device != H_dl.device or H_ul.dtype != H_dl.dtype:
            raise ValueError("DL and UL physical shape/device/dtype must agree.")
        if H_ul_est is None:
            H_ul_est = H_ul
        if H_ul_est.shape != H_ul.shape or H_ul_est.device != H_ul.device or H_ul_est.dtype != H_ul.dtype:
            raise ValueError("Public UL estimate must match the physical UL shape/device/dtype.")
        if not torch.equal(H_ul_est, H_ul):
            raise ValueError("This fixed experiment declares perfect UL CSI; estimator errors are out of scope.")
        protocol._complex_tensor(H_ul, "Physical UL")
        protocol._complex_tensor(standard_noise, "Standard physical noise", (H_dl.shape[0], 32, 256))
        if standard_noise.device != H_dl.device:
            raise ValueError("Standard physical noise must share the physical-channel device.")
        if standard_noise.requires_grad:
            raise ValueError("Standard physical noise must be nonlearned.")
        raw = self.ue(H_dl, gamma_db, self.layout.active_per_user)
        X, canonical_truth, energy = protocol.normalize_and_map(raw, self.layout)
        Y = protocol.transmit(H_ul, X, self.layout, N0=N0, standard_noise=standard_noise)
        recovery = protocol.recover(Y, H_ul_est, self.layout, N0=N0)
        H_ad, corrected, execution = self.decoder(
            recovery["z_hat"], recovery["reliability"], Y, H_ul_est, gamma_db)
        H_hat = inverse_angular_delay_transform(H_ad, 256)
        W_rzf = self.zf_solver(H_hat)
        power = W_rzf.abs().square().sum(dim=(1, 2), keepdim=True)
        W = W_rzf * torch.sqrt(1.0 / (power + 1e-9))
        result = None
        if diagnostics:
            # Simulator truth and realized noise power become diagnostic outputs
            # only after every decoder operation and deployed RZF are complete.
            result = {
                "z_hat": recovery["z_hat"].detach(), "z_corrected": corrected.detach(),
                "true_z": canonical_truth.detach(), "reliability": recovery["reliability"].detach(),
                "active_mask": self.decoder.active_mask.detach(), "energy": energy.detach(),
                "noise_power": (standard_noise * math.sqrt(N0 / 2.0)).abs().square().mean((1, 2)).detach(),
                "N0": N0, "lambda_rx": recovery["lambda_rx"].detach(),
                "gamma_db": float(gamma_db), "execution": execution,
            }
        return W, H_hat, result

    def forward_with_representation(self, H_dl, H_ul, gamma_db, *, standard_noise, H_ul_est=None):
        W, H_hat, _ = self._forward(H_dl, H_ul, gamma_db, standard_noise=standard_noise, H_ul_est=H_ul_est)
        return W, H_hat

    def forward_with_diagnostics(self, H_dl, H_ul, gamma_db, *, standard_noise, H_ul_est=None):
        W, H_hat, diagnostic = self._forward(
            H_dl, H_ul, gamma_db, standard_noise=standard_noise, H_ul_est=H_ul_est, diagnostics=True)
        return {"precoder": W, "channel_surrogate": H_hat, "diagnostics": diagnostic}

    def forward(self, H_dl, H_ul, gamma_db, *, standard_noise, H_ul_est=None):
        W, H_hat = self.forward_with_representation(
            H_dl, H_ul, gamma_db, standard_noise=standard_noise, H_ul_est=H_ul_est)
        return (W, H_hat) if self.training else W
