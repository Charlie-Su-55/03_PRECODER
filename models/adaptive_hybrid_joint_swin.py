"""Pure-AS, full-resolution Joint+Swin256 and its matched plain control.

This is an independent BS architecture. The UE, feedback channel, power rules and
deployed RZF are reused from 03, but no Full BS module is constructed. The joint
structure follows the audited pinned 08 decoder; there are no runtime 08 imports,
fake pilot observations, true-noise inputs, receive-power normalization or gates.
"""
from __future__ import annotations

from copy import deepcopy
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .adaptive_hybrid import AdaptiveHybridEncoder, AdaptiveHybridPrecoder, ConditionEmbed
from .adaptive_hybrid_as_receiver import regularized_recovery
from .adaptive_hybrid_as_spatial import AdaptiveHybridPrecoderASSpatial
from .baseline_common import inverse_angular_delay_transform
from .swin_precoder import SwinCFNetDecoder


ARCHITECTURE = "joint_swin256_v1"
ARMS = ("plain_swin256", "joint_swin256")
COMMON_BS_PREFIXES = ("decoder.codec.", "decoder.spatial_position.")
SETTINGS = {
    "width": 128,
    "heads": 4,
    "joint_layers": 2,
    "joint_ffn_dim": 256,
    "joint_ffn_activation": "GELU",
    "normalization": "pre-LayerNorm",
    "dropout": 0,
    "complex_symbols_per_user": 256,
    "observation_tokens": 256,
    "joint_tokens": 1024,
    "query_features": 15,
    "query_feature_order": [
        "real",
        "imag",
        "resource_coordinates_3",
        "slot_coordinates_3",
        "role_exclusive_superposed_2",
        "user_one_hot_4",
        "log1p_reliability_proxy"
    ],
    "reliability_feature": "log1p(clipped_diagonal_reliability_proxy); not calibrated precision",
    "observation_features": [
        "noisy_Y_ac",
        "H_ul_est",
        "resource_coordinates_3",
        "log1p_reliability_proxy_per_user",
        "feedback_snr_db/25"
    ],
    "coupling": "mean_resource(abs(h_k @ h_j.conj())**2/(norm2(h_k).clamp_min(1e-12)*norm2(h_j).clamp_min(1e-12)))",
    "self_attention_bias": "same_resource*overlap_head + channel_coupling*coupling_head",
    "observation_attention_bias": "same_resource*resource_cross_head",
    "head_bias_initial_value": 0.02,
    "residual_weight_std": 0.001,
    "spatial_cross_output_weight_std": 0.001,
    "extra_linear_weight_std": 0.02,
    "residual_gate": False,
    "receive_power_normalization": False,
    "codeword_real_order": "all real then all imaginary",
    "codeword_fc": [
        512,
        65536
    ],
    "patch_size": 4,
    "patch_grid": [
        64,
        8
    ],
    "full_delay_taps": 256,
    "antennas": 32,
    "spatial_tokens_per_user": 512,
    "spatial_cross_queries": 2048,
    "spatial_cross_keys": 1024,
    "spatial_position_features": "delay_coordinates_3+angle_coordinates_3+user_one_hot_4",
    "shared_spatial_position": True,
    "swin_blocks": 4,
    "swin_heads": 4,
    "window_size": 4,
    "swin_mlp_ratio": 4,
    "shifted_windows": False,
    "swin_implementation": "models.swin_precoder.SwinCFNetDecoder / SwinTransformerBlock",
    "inverse_transform": "models.baseline_common.inverse_angular_delay_transform",
    "attention_implementation": "torch.nn.functional.scaled_dot_product_attention",
    "query_chunk_size": 64,
    "attention_keys": "all keys in every query chunk",
    "activation_checkpoint": True,
    "checkpoint_use_reentrant": False,
    "checkpoint_scope": "joint layers and per-query attention projections/chunks; spatial attention; each Swin block",
    "recovery_lambda_floor": 0.000001,
    "recovery_solve_dtype": "complex64",
    "noise_estimator": "mean(abs(Y_ac)**2)/(1+10**(feedback_snr_db/10))",
    "noise_reduction": "all_batch_antennas_active_AS_symbols",
    "lambda_rule": "max(noise_est,lambda_floor)",
    "physical_slice": "K*d_f:K*d_f+d_a",
    "ue_power_ema": 0.02,
    "rzf_regularization": 0.001,
    "power_normalization": "per-subcarrier",
    "old_full_bs_modules": False,
    "common_initialization": "codec and spatial_position before RNG-isolated joint extras",
    "plain_skips": [
        "observation_memory",
        "joint_layers",
        "codeword_residual",
        "spatial_cross_attention"
    ]
}


def _coordinates(position, extent):
    scaled = position.float() / max(extent - 1, 1)
    return torch.stack((scaled, torch.sin(2 * math.pi * scaled),
                        torch.cos(2 * math.pi * scaled)), dim=-1)


def channel_coupling(H_ul_est):
    """Normalized channel overlap, averaged over observed physical resources."""
    h = H_ul_est.permute(0, 3, 1, 2)  # [B,T,K,Nt]
    gram = h @ h.conj().transpose(-2, -1)
    norm2 = h.abs().square().sum(dim=-1).clamp_min(1e-12)
    denominator = norm2.unsqueeze(-1) * norm2.unsqueeze(-2)
    return (gram.abs().square() / denominator).mean(dim=1)


def _run(function, *args, enabled):
    if enabled and torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


class ChunkedAttention(nn.Module):
    """Global all-key SDPA with bounded query chunks and inner checkpointing.

    K/V projections are computed once, inside the enclosing joint-layer/spatial
    checkpoint. Each query projection, chunk bias, SDPA and output projection is
    checkpointed independently: a math-SDPA fallback cannot retain all chunks'
    dense attention matrices. No dense full-query mask is constructed or cached.
    """

    def __init__(self, width=128, heads=4, query_chunk=64, symbols=256,
                 activation_checkpoint=True):
        super().__init__()
        if width % heads or query_chunk < 1:
            raise ValueError("Invalid attention width, heads or query chunk.")
        self.width, self.heads = width, heads
        self.query_chunk, self.symbols = query_chunk, symbols
        self.activation_checkpoint = activation_checkpoint
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def _split(self, tensor):
        batch, tokens, _ = tensor.shape
        return tensor.reshape(batch, tokens, self.heads, self.width // self.heads).transpose(1, 2)

    def _chunk(self, query, key, value, coupling, overlap_weight,
               coupling_weight, resource_weight, start, bias_kind):
        q = self._split(self.q(query))
        bias = None
        if bias_kind != "none":
            qi = torch.arange(start, start + query.shape[1], device=query.device)
            ki = torch.arange(key.shape[-2], device=query.device)
            same_resource = (qi.remainder(self.symbols)[:, None]
                             == ki.remainder(self.symbols)[None, :])
            same_resource = same_resource.to(dtype=query.dtype)
            if bias_kind == "self":
                per_user = coupling[:, qi.div(self.symbols, rounding_mode="floor")]
                per_user = per_user[:, :, ki.div(self.symbols, rounding_mode="floor")]
                bias = (same_resource[None, None] * overlap_weight[None, :, None, None]
                        + per_user[:, None] * coupling_weight[None, :, None, None])
            elif bias_kind == "observation":
                bias = same_resource[None, None] * resource_weight[None, :, None, None]
            else:
                raise ValueError("Unknown attention bias kind.")
        attended = F.scaled_dot_product_attention(
            q, key, value, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.width)
        return self.out(attended)

    def forward(self, query, memory, coupling=None, overlap_weight=None,
                coupling_weight=None, resource_weight=None, bias_kind="none"):
        key, value = self._split(self.k(memory)), self._split(self.v(memory))
        chunks = []
        for start in range(0, query.shape[1], self.query_chunk):
            chunks.append(_run(
                self._chunk, query[:, start:start + self.query_chunk], key, value,
                coupling, overlap_weight, coupling_weight, resource_weight,
                start, bias_kind,
                enabled=self.activation_checkpoint and self.training,
            ))
        return torch.cat(chunks, dim=1)


class ObservationMemory(nn.Module):
    """Exactly the received AS observations and receiver-available features."""

    def __init__(self):
        super().__init__()
        self.y = nn.Linear(64, 128)
        self.channel = nn.Linear(256, 128)
        self.position = nn.Linear(3, 128)
        self.reliability = nn.Linear(4, 128)
        self.snr = nn.Linear(1, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, Y_ac, H_ul_est, reliability, resource_coordinates, snr_feature):
        batch = Y_ac.shape[0]
        y = torch.view_as_real(Y_ac.transpose(1, 2).contiguous()).flatten(-2)
        h = torch.view_as_real(H_ul_est.permute(0, 3, 1, 2).contiguous()).flatten(-3)
        memory = (self.y(y) + self.channel(h)
                  + self.position(resource_coordinates).unsqueeze(0)
                  + self.reliability(torch.log1p(reliability))
                  + self.snr(snr_feature.expand(batch, 256, 1)))
        return self.norm(memory)


class JointLayer(nn.Module):
    def __init__(self, query_chunk, activation_checkpoint):
        super().__init__()
        self.self_norm = nn.LayerNorm(128)
        self.self_attention = ChunkedAttention(query_chunk=query_chunk,
                                              activation_checkpoint=activation_checkpoint)
        self.cross_norm = nn.LayerNorm(128)
        self.memory_norm = nn.LayerNorm(128)
        self.cross_attention = ChunkedAttention(query_chunk=query_chunk,
                                               activation_checkpoint=activation_checkpoint)
        self.mlp_norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, x, memory, coupling, overlap_weight, coupling_weight, resource_weight):
        normalized = self.self_norm(x)
        x = x + self.self_attention(
            normalized, normalized, coupling, overlap_weight, coupling_weight,
            resource_weight, "self"
        )
        x = x + self.cross_attention(
            self.cross_norm(x), self.memory_norm(memory), coupling, overlap_weight,
            coupling_weight, resource_weight, "observation"
        )
        return x + self.mlp(self.mlp_norm(x))


def _initialize_extra(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class JointSwin256Decoder(nn.Module):
    """Full 256x32 shared Swin decoder, with optional global joint processing."""

    def __init__(self, arm, query_chunk=64, activation_checkpoint=True):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unsupported Joint+Swin256 arm.")
        self.arm = arm
        self.activation_checkpoint = activation_checkpoint
        # Common modules MUST precede all joint-only modules. Both arms therefore
        # have identical common tensors under the same construction seed.
        self.codec = SwinCFNetDecoder(SimpleNamespace(
            SWIN_NC=256, ANTENNAS=32, SWIN_PATCH_SIZE=4, SWIN_WINDOW_SIZE=4,
            SWIN_EMBED_DIM=128, SWIN_NUM_HEADS=4, SWIN_MLP_RATIO=4.0,
            SWIN_NUM_BLOCKS=4, SWIN_CODEWORD_DIM=512,
        ))
        self.spatial_position = nn.Linear(10, 128)
        self.spatial_position.apply(_initialize_extra)
        resources = _coordinates(torch.arange(256), 256)
        one_hot = torch.eye(4)
        rows = _coordinates(torch.arange(64), 64)
        columns = _coordinates(torch.arange(8), 8)
        spatial = torch.cat((
            rows[None, :, None, :].expand(4, 64, 8, 3),
            columns[None, None, :, :].expand(4, 64, 8, 3),
            one_hot[:, None, None, :].expand(4, 64, 8, 4),
        ), dim=-1).reshape(4, 512, 10)
        self.register_buffer("resource_coordinates", resources, persistent=False)
        self.register_buffer("spatial_coordinates", spatial, persistent=False)
        if arm == "joint_swin256":
            # Extra initialization must not alter the RNG state seen after
            # common construction (including the first training feedback noise).
            with torch.random.fork_rng(devices=[]):
                self.observation_memory = ObservationMemory()
                self.query_projection = nn.Linear(15, 128)
                self.joint_layers = nn.ModuleList([
                    JointLayer(query_chunk, activation_checkpoint) for _ in range(2)
                ])
                self.overlap_bias = nn.Parameter(torch.full((4,), 0.02))
                self.coupling_bias = nn.Parameter(torch.full((4,), 0.02))
                self.resource_cross_bias = nn.Parameter(torch.full((4,), 0.02))
                self.residual_norm = nn.LayerNorm(128)
                self.residual_head = nn.Linear(128, 2)
                self.spatial_query_norm = nn.LayerNorm(128)
                self.spatial_context_norm = nn.LayerNorm(128)
                self.spatial_attention = ChunkedAttention(
                    query_chunk=query_chunk, activation_checkpoint=activation_checkpoint
                )
                for module in (self.observation_memory, self.query_projection, self.joint_layers,
                               self.residual_head, self.spatial_attention):
                    module.apply(_initialize_extra)
                nn.init.normal_(self.residual_head.weight, std=0.001)
                nn.init.normal_(self.spatial_attention.out.weight, std=0.001)
            roles = torch.tensor([0.0, 1.0]).expand(4, 256, 2)
            fixed = torch.cat((
                resources[None].expand(4, 256, 3),
                resources[None].expand(4, 256, 3), roles,
                one_hot[:, None, :].expand(4, 256, 4),
            ), dim=-1)
            self.register_buffer("query_geometry", fixed, persistent=False)

    def _spatial_cross(self, spatial, context):
        return self.spatial_attention(
            self.spatial_query_norm(spatial), self.spatial_context_norm(context)
        )

    def forward(self, z_hat, reliability, Y_ac, H_ul_est, snr_db):
        if z_hat.ndim != 3 or tuple(z_hat.shape[1:]) != (256, 4):
            raise ValueError("Expected recovered codewords [B,256,4].")
        if reliability.shape != z_hat.shape:
            raise ValueError("Reliability must match recovered codewords.")
        if tuple(Y_ac.shape[1:]) != (32, 256) or tuple(H_ul_est.shape[1:]) != (4, 32, 256):
            raise ValueError("Receiver observations must cover the full pure-AS physical slice.")
        if Y_ac.shape[0] != z_hat.shape[0] or H_ul_est.shape[0] != z_hat.shape[0]:
            raise ValueError("Receiver observation batch sizes must agree.")
        snr = float(snr_db.detach().item()) if torch.is_tensor(snr_db) else float(snr_db)
        if not math.isfinite(snr):
            raise ValueError("Feedback SNR must be finite.")
        batch = z_hat.shape[0]
        enabled = self.activation_checkpoint and self.training
        corrected = z_hat.permute(0, 2, 1)
        context = None
        if self.arm == "joint_swin256":
            snr_feature = z_hat.real.new_full((1, 1, 1), snr / 25.0)
            memory = _run(self.observation_memory, Y_ac, H_ul_est, reliability,
                          self.resource_coordinates, snr_feature, enabled=enabled)
            features = torch.cat((
                torch.view_as_real(corrected.contiguous()),
                self.query_geometry.unsqueeze(0).expand(batch, -1, -1, -1),
                torch.log1p(reliability.permute(0, 2, 1)).unsqueeze(-1),
            ), dim=-1).reshape(batch, 1024, 15)
            context = self.query_projection(features)
            coupling = channel_coupling(H_ul_est)
            for layer in self.joint_layers:
                context = _run(layer, context, memory, coupling, self.overlap_bias,
                               self.coupling_bias, self.resource_cross_bias, enabled=enabled)
            residual = self.residual_head(self.residual_norm(context)).reshape(batch, 4, 256, 2)
            corrected = corrected + torch.complex(residual[..., 0], residual[..., 1])
        # Important: all real symbols followed by all imaginary symbols, not
        # interleaved real/imag triples and not the former 32-delay expansion.
        codeword = torch.cat((corrected.real, corrected.imag), dim=-1)
        spatial = self.codec.fc(codeword.reshape(batch * 4, 512)).reshape(batch, 4, 512, 128)
        spatial = spatial + self.spatial_position(self.spatial_coordinates).unsqueeze(0)
        if context is not None:
            spatial_flat = spatial.reshape(batch, 2048, 128)
            spatial = (spatial_flat + _run(self._spatial_cross, spatial_flat, context,
                                          enabled=enabled)).reshape(batch, 4, 512, 128)
        x = spatial.reshape(batch * 4, 512, 128)
        for block in self.codec.blocks:
            x = _run(block, x, 64, 8, enabled=enabled)
        x = self.codec.final_proj(self.codec.norm(x))
        x = x.reshape(batch * 4, 64, 8, 2, 4, 4)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        H_ad = x.reshape(batch, 4, 2, 256, 32)
        # Runtime shapes only, not fabricated execution flags. Preflight uses
        # actual module forward hooks to prove the joint operations execute.
        execution = {
            "recovered_shape": list(z_hat.shape),
            "joint_context_shape": None if context is None else list(context.shape),
            "spatial_shape": list(spatial.shape), "angular_delay_shape": list(H_ad.shape),
        }
        return H_ad, corrected.permute(0, 2, 1), execution


class AdaptiveHybridPrecoderJointSwin(nn.Module):
    """Independent full-resolution BS, original physical path and trainable UE."""

    add_noise = AdaptiveHybridPrecoder.add_noise
    transmit = AdaptiveHybridPrecoder.transmit
    zf_solver = AdaptiveHybridPrecoder.zf_solver
    # These methods only inspect explicit UE keys and strictly merge them. They
    # do not construct or execute the old spatial decoder or any Full BS module.
    ue_state_keys = AdaptiveHybridPrecoderASSpatial.ue_state_keys
    initialize_ue_from_full = AdaptiveHybridPrecoderASSpatial.initialize_ue_from_full
    ue_parameters = AdaptiveHybridPrecoderASSpatial.ue_parameters
    bs_parameters = AdaptiveHybridPrecoderASSpatial.bs_parameters

    def __init__(self, cfg):
        super().__init__()
        expected = {"K_USERS": 4, "ANTENNAS": 32, "SUBCARRIERS": 256,
                    "NUM_SUBBANDS": 4, "D_MODEL": 256, "D_F_MAX_PER_USER": 64,
                    "D_A_MAX_SHARED": 256, "NUM_ENCODER_LAYERS": 4, "NUM_HEADS": 8}
        for name, value in expected.items():
            if getattr(cfg, name, None) != value:
                raise ValueError(f"Joint+Swin256 fixes {name}={value}.")
        if tuple(tuple(pair) for pair in cfg.GAMMA_GRID) != ((0, 256),):
            raise ValueError("Joint+Swin256 requires singleton pure-AS [(0,256)].")
        if not math.isfinite(float(cfg.TOTAL_POWER)) or float(cfg.TOTAL_POWER) <= 0:
            raise ValueError("TOTAL_POWER must be positive and finite.")
        if getattr(cfg, "arm", None) not in ARMS:
            raise ValueError("Joint+Swin256 requires a declared matched experimental arm.")
        fixed = {
            "JOINT_SWIN_LAMBDA_FLOOR": 1e-6, "JOINT_SWIN_SOLVE_DTYPE": "complex64",
            "JOINT_SWIN_QUERY_CHUNK": 64, "JOINT_SWIN_ACTIVATION_CHECKPOINT": True,
        }
        for name, value in fixed.items():
            actual = getattr(cfg, name, value)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f"Joint+Swin256 fixes {name}={value!r}.")
        self.cfg, self.arm = cfg, cfg.arm
        self.K, self.Nt, self.Nsc, self.N_sb, self.D = 4, 32, 256, 4, 256
        self.lambda_floor, self.solve_dtype = 1e-6, "complex64"
        self.ue_frozen = False
        self.encoder = AdaptiveHybridEncoder(cfg)
        self.cond_embed = ConditionEmbed(256, 64, 256)
        self.register_buffer("running_pwr", torch.ones(1))
        self.pwr_momentum = 0.02
        self.decoder = JointSwin256Decoder(
            self.arm, query_chunk=64, activation_checkpoint=True
        )

    @property
    def architecture(self):
        return ARCHITECTURE

    @property
    def architecture_metadata(self):
        return {
            "architecture": self.architecture, "model_class": type(self).__name__,
            "arm": self.arm, "ue_frozen": False, "ue_class": "AdaptiveHybridEncoder",
            "ue_ffn_dim": 2048, "ue_layers_per_branch": 4, "D_MODEL": 256,
            "allocation_grid": [[0, 256]], "max_dimensions": [64, 256],
            "running_pwr_shape": [1], "cfg_idx": 0,
            "ue_power": "original batch scale and EMA in train; frozen EMA in eval",
            "common_bs_prefixes": list(COMMON_BS_PREFIXES),
            "settings": deepcopy(SETTINGS),
            "diagnostics": "explicit return only, detached truth after H_hat and W",
        }

    def _forward(self, H_dl, H_ul, snr_db, d_f, d_a, cfg_idx,
                 H_ul_est=None, *, with_diagnostics=False):
        if (d_f, d_a, cfg_idx) != (0, 256, 0):
            raise ValueError("Joint+Swin256 supports only (d_f,d_a,cfg_idx)=(0,256,0).")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (d_f, d_a, cfg_idx)):
            raise TypeError("Allocation and cfg_idx must be integers.")
        if H_dl.ndim != 4 or tuple(H_dl.shape[1:]) != (4, 32, 256):
            raise ValueError("Expected H_dl[B,4,32,256].")
        if H_ul.shape != H_dl.shape or H_ul.device != H_dl.device or H_ul.dtype != H_dl.dtype:
            raise ValueError("Physical DL/UL tensors must share shape, device and dtype.")
        if H_dl.dtype != torch.complex64:
            raise TypeError("Joint+Swin256 requires complex64 physical channels.")
        if H_ul_est is None:
            H_ul_est = H_ul
        if H_ul_est.shape != H_ul.shape or H_ul_est.device != H_ul.device:
            raise ValueError("H_ul_est must match H_ul shape and device.")
        if H_ul_est.dtype != H_ul.dtype:
            raise TypeError("H_ul_est must match H_ul dtype.")
        batch = H_dl.shape[0]
        cond = self.cond_embed(d_f, d_a, snr_db, H_dl.device)
        z_f_flat, z_a_flat = self.encoder(
            H_dl.reshape(batch * self.K, self.Nt, self.Nsc), 0, 64, cond
        )
        n_sym = z_f_flat.shape[1] + z_a_flat.shape[1]
        energy = z_f_flat.abs().square().sum() + z_a_flat.abs().square().sum()
        pwr_batch = energy / (batch * self.K * n_sym) + 1e-9
        if self.training:
            with torch.no_grad():
                self.running_pwr[cfg_idx] = (
                    (1 - self.pwr_momentum) * self.running_pwr[cfg_idx]
                    + self.pwr_momentum * pwr_batch.detach()
                )
            scale = torch.sqrt(pwr_batch)
        else:
            scale = torch.sqrt(self.running_pwr[cfg_idx] + 1e-9)
        Z_f = (z_f_flat / scale).reshape(batch, self.K, -1)
        Z_a = (z_a_flat / scale).reshape(batch, self.K, -1)
        _, Y_ac = self.transmit(Z_f, Z_a, H_ul, snr_db, d_f, d_a)
        recovery = regularized_recovery(
            Y_ac, H_ul_est, snr_db, d_f, d_a,
            lambda_floor=self.lambda_floor, solve_dtype=self.solve_dtype,
        )
        H_ad, corrected, execution = self.decoder(
            recovery["z_hat"], recovery["reliability"], Y_ac, H_ul_est, snr_db
        )
        H_hat = inverse_angular_delay_transform(H_ad, self.Nsc)
        W_zf = self.zf_solver(H_hat)
        frob = W_zf.abs().square().sum(dim=(1, 2), keepdim=True)
        W = W_zf * torch.sqrt(self.cfg.TOTAL_POWER / (frob + 1e-9))
        diagnostics = None
        if with_diagnostics:
            # True transmitted codewords are exposed only AFTER Hhat and W.
            # They are never input to a learned BS module or retained on self.
            diagnostics = {
                "z_hat": recovery["z_hat"].detach(),
                "z_corrected": corrected.detach(), "true_z_a": Z_a.detach(),
                "reliability": recovery["reliability"].detach(),
                "noise_est": recovery["noise_est"].detach(),
                "regularization": recovery["regularization"].detach(),
                "feedback_snr_db": float(snr_db.detach().item()) if torch.is_tensor(snr_db) else float(snr_db),
                "allocation_df": d_f, "allocation_da": d_a, "execution": execution,
            }
        return W, H_hat, diagnostics

    def forward_with_representation(self, H_dl, H_ul, snr_db, d_f, d_a, cfg_idx,
                                    H_ul_est=None):
        W, H_hat, _ = self._forward(H_dl, H_ul, snr_db, d_f, d_a, cfg_idx, H_ul_est)
        return W, H_hat

    def forward_with_diagnostics(self, H_dl, H_ul, snr_db, d_f, d_a, cfg_idx,
                                H_ul_est=None):
        W, H_hat, diagnostics = self._forward(
            H_dl, H_ul, snr_db, d_f, d_a, cfg_idx, H_ul_est, with_diagnostics=True
        )
        return {"precoder": W, "channel_surrogate": H_hat, "diagnostics": diagnostics}

    def forward(self, H_dl, H_ul, snr_db, d_f, d_a, cfg_idx, H_ul_est=None):
        W, H_hat = self.forward_with_representation(
            H_dl, H_ul, snr_db, d_f, d_a, cfg_idx, H_ul_est
        )
        return (W, H_hat) if self.training else W


def initialize_ue_from_full(model, source_state):
    if not isinstance(model, AdaptiveHybridPrecoderJointSwin):
        raise TypeError("Expected AdaptiveHybridPrecoderJointSwin.")
    return model.initialize_ue_from_full(source_state)
