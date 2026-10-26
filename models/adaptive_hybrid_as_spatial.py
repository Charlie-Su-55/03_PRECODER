"""Independent pure-AS spatial BS decoder; the original Full files are untouched.

No Full Stage-1, context, refinement, channel head or residual shortcut is
constructed. Only the UE, condition embedding, physical feedback operators and
normalized deployed RZF convention are reused. True codewords are returned only
by an explicit offline diagnostic API, after reconstruction and precoding.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import math

import torch
from torch import nn

from .adaptive_hybrid import AdaptiveHybridEncoder, AdaptiveHybridPrecoder, ConditionEmbed
from .adaptive_hybrid_as_receiver import regularized_recovery
from .baseline_common import inverse_angular_delay_transform
from .swin_precoder import SwinTransformerBlock


UE_STATE_PREFIXES = ("encoder.", "cond_embed.")
UE_STATE_BUFFER = "running_pwr"


class ASSpatialDecoder(nn.Module):
    """Ordered codeword -> learned spatial expansion -> existing Swin blocks."""

    def __init__(self):
        super().__init__()
        self.symbols = 256
        self.delay_taps = 32
        self.antennas = 32
        self.patch_size = 4
        self.width = 128
        self.grid_height = self.grid_width = 8
        self.input_features = self.symbols * 3 + 1
        # An actual learned mapping, not an identification of codeword indices
        # with angle/delay coordinates. Output index is the spatial patch index.
        self.spatial_expansion = nn.Linear(self.input_features, 64 * self.width)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(128, 4, 4, 4.0) for _ in range(4)
        ])
        self.norm = nn.LayerNorm(self.width)
        self.final_proj = nn.Linear(self.width, 2 * self.patch_size**2)

    def forward(self, z_hat, reliability, snr_db):
        """Only recovered receiver data enter this decoder.

        z_hat/reliability: [B,256,K], preserving the original subband-prefix
        serialization. The ordered per-user triples are flattened without
        relabeling them as physical coordinates; SNR/25 is one extra scalar.
        """
        if z_hat.ndim != 3 or z_hat.shape[1] != self.symbols:
            raise ValueError("Spatial decoder requires z_hat[B,256,K].")
        if reliability.shape != z_hat.shape:
            raise ValueError("Reliability must match recovered codeword shape.")
        batch, _, users = z_hat.shape
        features = torch.stack((z_hat.real, z_hat.imag, reliability), dim=-1)
        features = features.permute(0, 2, 1, 3).reshape(batch * users, -1)
        snr = float(snr_db.detach().item()) if torch.is_tensor(snr_db) else float(snr_db)
        if not math.isfinite(snr):
            raise ValueError("Feedback SNR must be finite.")
        snr_feature = features.new_full((batch * users, 1), snr / 25.0)
        features = torch.cat((features, snr_feature), dim=-1)
        x = self.spatial_expansion(features).reshape(batch * users, 64, self.width)
        for block in self.blocks:
            x = block(x, self.grid_height, self.grid_width)
        x = self.final_proj(self.norm(x))
        p = self.patch_size
        # Exactly the existing Swin decoder patch unembedding convention:
        # [delay_patch,angle_patch,Re/Im,delay_in_patch,angle_in_patch].
        x = x.reshape(batch * users, 8, 8, 2, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.reshape(batch, users, 2, self.delay_taps, self.antennas)


class AdaptiveHybridPrecoderASSpatial(nn.Module):
    """Pure-AS UE feedback plus a completely new shared spatial BS decoder."""

    # Reuse the exact established physical/noise/RZF implementations; no Full
    # constructor is called and no old BS parameters or modules are retained.
    add_noise = AdaptiveHybridPrecoder.add_noise
    transmit = AdaptiveHybridPrecoder.transmit
    zf_solver = AdaptiveHybridPrecoder.zf_solver

    def __init__(self, cfg):
        super().__init__()
        expected = {"K_USERS": 4, "ANTENNAS": 32, "SUBCARRIERS": 256,
                    "NUM_SUBBANDS": 4, "D_MODEL": 256,
                    "D_F_MAX_PER_USER": 64, "D_A_MAX_SHARED": 256,
                    "NUM_ENCODER_LAYERS": 4, "NUM_HEADS": 8}
        for name, value in expected.items():
            if getattr(cfg, name, None) != value:
                raise ValueError(f"as_spatial_v1 fixes {name}={value}.")
        if tuple(tuple(pair) for pair in cfg.GAMMA_GRID) != ((0, 256),):
            raise ValueError("as_spatial_v1 requires singleton pure-AS [(0,256)].")
        if not math.isfinite(float(cfg.TOTAL_POWER)) or float(cfg.TOTAL_POWER) <= 0:
            raise ValueError("TOTAL_POWER must be positive and finite.")
        arm = getattr(cfg, "arm", None)
        if arm not in ("frozen_ue", "joint"):
            raise ValueError("ASSpatial arm must be 'frozen_ue' or 'joint'.")
        floor = getattr(cfg, "AS_SPATIAL_LAMBDA_FLOOR", 1e-6)
        dtype = getattr(cfg, "AS_SPATIAL_SOLVE_DTYPE", "complex64")
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor <= 0:
            raise ValueError("AS spatial lambda floor must be positive and finite.")
        if dtype != "complex64":
            raise ValueError("as_spatial_v1 fixes receiver solve precision to complex64.")
        self.cfg = cfg
        self.K, self.Nt, self.Nsc, self.N_sb, self.D = 4, 32, 256, 4, 256
        self.arm = arm
        self.lambda_floor = float(floor)
        self.solve_dtype = dtype
        self.encoder = AdaptiveHybridEncoder(cfg)
        self.cond_embed = ConditionEmbed(self.D, 64, 256)
        self.register_buffer("running_pwr", torch.ones(1))
        self.pwr_momentum = 0.02
        self.decoder = ASSpatialDecoder()
        self.set_ue_frozen(arm == "frozen_ue")

    @property
    def architecture(self):
        return "as_spatial_v1"

    @property
    def ue_state_keys(self):
        return tuple(name for name in self.state_dict()
                     if name.startswith(UE_STATE_PREFIXES) or name == UE_STATE_BUFFER)

    @property
    def architecture_metadata(self):
        return {
            "architecture": self.architecture, "model_class": type(self).__name__,
            "arm": self.arm, "ue_frozen": self.ue_frozen,
            "ue_class": "AdaptiveHybridEncoder", "ue_ffn_dim": 2048,
            "ue_layers_per_branch": 4, "D_MODEL": 256,
            "allocation_grid": [[0, 256]], "max_dimensions": [64, 256],
            "running_pwr_shape": [1], "cfg_idx": 0,
            "frozen_ue_power": "source running_pwr, no updates",
            "joint_ue_power": "original batch scale and EMA in train; frozen EMA in eval",
            "receiver": {
                "function": "adaptive_hybrid_as_receiver.regularized_recovery",
                "lambda_floor": self.lambda_floor, "solve_dtype": self.solve_dtype,
                "noise_estimator": "mean(abs(Y_ac)**2)/(1+10**(feedback_snr_db/10))",
                "noise_reduction": "all_batch_antennas_active_AS_symbols",
                "lambda_rule": "max(noise_est,lambda_floor)",
                "physical_slice": "K*d_f:K*d_f+d_a",
            },
            "decoder": {
                "shared_across_users": True, "ordered_complex_symbols": 256,
                "input": "ordered per-symbol real/imag/reliability triples plus SNR/25",
                "input_features": 769, "spatial_expansion": [769, 8192],
                "patch_grid": [8, 8], "angular_delay_shape": [32, 32],
                "patch_size": 4, "width": 128, "heads": 4, "window_size": 4,
                "swin_blocks": 4, "mlp_ratio": 4.0, "dropout": 0.0,
                "shifted_windows": False,
                "block_implementation": "models.swin_precoder.SwinTransformerBlock",
                "initialization": "new random BS; no imported Full/Swin BS weights",
                "inverse_transform": "models.baseline_common.inverse_angular_delay_transform",
            },
            "rzf_regularization": 1e-3, "power_normalization": "per-subcarrier",
            "old_stage1_or_refinement_or_head": False,
            "diagnostics": "explicit return only, detached truth after H_hat and W",
        }

    def ue_parameters(self):
        yield from self.encoder.parameters()
        yield from self.cond_embed.parameters()

    def bs_parameters(self):
        yield from self.decoder.parameters()

    def set_ue_frozen(self, frozen):
        if type(frozen) is not bool:
            raise TypeError("frozen must be a bool.")
        self.ue_frozen = frozen
        for parameter in self.ue_parameters():
            parameter.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)
        self.cond_embed.train(self.training and not frozen)
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "ue_frozen", False):
            self.encoder.eval()
            self.cond_embed.eval()
        return self

    def initialize_ue_from_full(self, source_state):
        """Validate all imported UE tensors before strict loading; leave BS fresh."""
        if not isinstance(source_state, Mapping):
            raise TypeError("Source must be the Full tensor state_dict.")
        target = self.state_dict()
        expected = set(self.ue_state_keys)
        supplied = {name for name in source_state
                    if isinstance(name, str) and
                    (name.startswith(UE_STATE_PREFIXES) or name == UE_STATE_BUFFER)}
        if supplied != expected:
            raise ValueError(f"UE source key mismatch: missing={sorted(expected-supplied)}, unexpected={sorted(supplied-expected)}")
        for name in sorted(expected):
            value = source_state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name}: expected a tensor.")
            if value.shape != target[name].shape:
                raise ValueError(f"{name}: UE source shape mismatch.")
            if value.dtype != target[name].dtype:
                raise TypeError(f"{name}: UE source dtype mismatch.")
        merged = OrderedDict((name, source_state[name] if name in expected else value)
                             for name, value in target.items())
        if hasattr(target, "_metadata"):
            merged._metadata = deepcopy(target._metadata)
            for name, value in getattr(source_state, "_metadata", {}).items():
                if name in ("encoder", "cond_embed") or name.startswith(UE_STATE_PREFIXES):
                    merged._metadata[name] = deepcopy(value)
        return self.load_state_dict(merged, strict=True)

    def _forward(self, H_dl, H_ul, snr_db, d_f, d_a, cfg_idx,
                 H_ul_est=None, *, with_diagnostics=False):
        if (d_f, d_a, cfg_idx) != (0, 256, 0):
            raise ValueError("ASSpatial supports only (d_f,d_a,cfg_idx)=(0,256,0).")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (d_f, d_a, cfg_idx)):
            raise TypeError("Allocation and cfg_idx must be integers.")
        if H_dl.ndim != 4 or tuple(H_dl.shape[1:]) != (4, 32, 256):
            raise ValueError("Expected H_dl[B,4,32,256].")
        if H_ul.shape != H_dl.shape or H_ul.device != H_dl.device or H_ul.dtype != H_dl.dtype:
            raise ValueError("Physical DL/UL tensors must share shape, device and dtype.")
        if H_dl.dtype != torch.complex64:
            raise TypeError("as_spatial_v1 requires complex64 physical channels.")
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
        # Exact original normalization equations, with one explicit arm policy:
        # frozen UE uses the archived inference scale even while the BS trains.
        n_sym = z_f_flat.shape[1] + z_a_flat.shape[1]
        energy = (z_f_flat.abs() ** 2).sum() + (z_a_flat.abs() ** 2).sum()
        pwr_batch = energy / (batch * self.K * n_sym) + 1e-9
        if self.training and not self.ue_frozen:
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
        H_ad = self.decoder(recovery["z_hat"], recovery["reliability"], snr_db)
        H_hat = inverse_angular_delay_transform(H_ad, self.Nsc)
        W_zf = self.zf_solver(H_hat)
        frob = (W_zf.abs() ** 2).sum(dim=(1, 2), keepdim=True)
        W = W_zf * torch.sqrt(self.cfg.TOTAL_POWER / (frob + 1e-9))
        diagnostics = None
        if with_diagnostics:
            # True transmitted codewords become visible only here, AFTER every
            # learned receiver/decoder operation. Never stored on the model.
            diagnostics = {
                "z_hat": recovery["z_hat"].detach(),
                "true_z_a": Z_a.detach(),
                "reliability": recovery["reliability"].detach(),
                "noise_est": recovery["noise_est"].detach(),
                "regularization": recovery["regularization"].detach(),
                "feedback_snr_db": float(snr_db.detach().item()) if torch.is_tensor(snr_db) else float(snr_db),
                "allocation_df": d_f, "allocation_da": d_a,
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
    """Functional companion to the strict UE-only instance initializer."""
    if not isinstance(model, AdaptiveHybridPrecoderASSpatial):
        raise TypeError("Expected AdaptiveHybridPrecoderASSpatial.")
    return model.initialize_ue_from_full(source_state)
