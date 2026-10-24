"""Isolated BS-only AS message residual; the original model is unchanged.

The GNN and refinement forwards deliberately retain the original operations.
Only the already-computed per-user message is additionally returned, then
injected after refinement when the physical AS allocation is active. The top
forward differs only in explicitly passing that allocation activity. Keeping
these local forwards avoids hooks, caches, and changes to the original API.
"""

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn

from .adaptive_hybrid import AdaptiveHybridPrecoder


def _require_child_only_state(module, expected_children):
    """Fail loudly if a later original implementation needs a new adapter."""
    if set(module._modules) != set(expected_children):
        raise RuntimeError(f"Unsupported {type(module).__name__} child modules.")
    if module._parameters or module._buffers:
        raise RuntimeError(
            f"Unsupported direct parameter/buffer on {type(module).__name__}."
        )


class _ASDirectAggregation(nn.Module):
    """Re-register initialized modules, without drawing new random weights."""

    def __init__(self, original):
        super().__init__()
        _require_child_only_state(
            original, ("msg_proj", "user_query_attn", "pair_agg")
        )
        self.d_model = original.d_model
        self.d_a = original.d_a
        for name, module in original.named_children():
            self.add_module(name, module)
        self.training = original.training

    def forward(self, Y_ac_feat, uv_feat):
        B, N_sb, Nt, D = Y_ac_feat.shape
        K = uv_feat.shape[2]

        Y_ac_flat = Y_ac_feat.reshape(B * N_sb, Nt, D)
        uv_flat = uv_feat.reshape(B * N_sb, K, D)

        user_msg, _ = self.user_query_attn(uv_flat, Y_ac_flat, Y_ac_flat)
        msg = self.msg_proj(user_msg)

        Y_i = Y_ac_flat.unsqueeze(2).expand(B * N_sb, Nt, Nt, D)
        Y_j = Y_ac_flat.unsqueeze(1).expand(B * N_sb, Nt, Nt, D)

        msg_sum = msg.sum(dim=1, keepdim=True).unsqueeze(2)
        Y_i = Y_i + msg_sum
        Y_j = Y_j + msg_sum

        A_feat = self.pair_agg(torch.cat([Y_i, Y_j], dim=-1))
        A_feat = A_feat.reshape(B, N_sb, Nt, Nt, self.d_a)
        return A_feat, msg.reshape(B, N_sb, K, D)


class _ASDirectRefinement(nn.Module):
    """Original initialized refinement modules plus one zero scalar gate."""

    def __init__(self, original):
        super().__init__()
        _require_child_only_state(
            original,
            ("uv_estimator", "A_estimator", "w_solver_attn", "A_to_bias",
             "cross_user_attn", "norm_cross", "ffn", "norm1", "norm2", "norm3"),
        )
        for name in ("K", "Nt", "N_sb", "d_model", "d_a", "n_heads"):
            setattr(self, name, getattr(original, name))
        for name, module in original.named_children():
            self.add_module(
                name, _ASDirectAggregation(module) if name == "A_estimator" else module
            )
        reference = next(original.parameters())
        self.alpha = nn.Parameter(reference.new_zeros(()))
        self.training = original.training

    def forward(self, W_feat, Y_f_feat, Y_ac_feat, *, as_active):
        B, K, Nt, N_sb, D = W_feat.shape
        H = self.n_heads

        W_pool = W_feat.mean(dim=2)
        Yf_pool = Y_f_feat.mean(dim=2)
        uv_input = torch.cat([W_pool, Yf_pool], dim=-1)
        uv_feat = self.uv_estimator(uv_input)
        uv_feat = uv_feat.permute(0, 2, 1, 3).contiguous()

        Y_ac_perm = Y_ac_feat.permute(0, 2, 1, 3).contiguous()
        A_feat, msg = self.A_estimator(Y_ac_perm, uv_feat)
        attn_bias = self.A_to_bias(A_feat)
        attn_bias = attn_bias.permute(0, 1, 4, 2, 3).contiguous()

        attn_bias_K = attn_bias.unsqueeze(2).expand(B, N_sb, K, H, Nt, Nt).contiguous()
        attn_bias_K = attn_bias_K.reshape(B * N_sb * K * H, Nt, Nt)

        W_perm = W_feat.permute(0, 3, 1, 2, 4).contiguous()
        W_flat = W_perm.reshape(B * N_sb * K, Nt, D)

        attn_out, _ = self.w_solver_attn(
            W_flat, W_flat, W_flat, attn_mask=attn_bias_K
        )
        W_flat = self.norm1(W_flat + attn_out)

        Y_f_perm = Y_f_feat.permute(0, 3, 1, 2, 4).contiguous()
        Y_f_flat = Y_f_perm.reshape(B * N_sb * K, Nt, D)
        W_flat = self.norm2(W_flat + Y_f_flat)

        W_after_solver = W_flat.reshape(B, N_sb, K, Nt, D)
        W_for_cross = W_after_solver.permute(0, 1, 3, 2, 4).contiguous()
        W_for_cross = W_for_cross.reshape(B * N_sb * Nt, K, D)

        cross_out, _ = self.cross_user_attn(W_for_cross, W_for_cross, W_for_cross)
        W_after_cross = cross_out.reshape(B, N_sb, Nt, K, D).permute(0, 1, 3, 2, 4).contiguous()

        W_combined = (W_after_solver + W_after_cross).reshape(B * N_sb * K, Nt, D)
        W_flat = self.norm_cross(W_combined)

        W_flat = self.norm3(W_flat + self.ffn(W_flat))

        W_out = W_flat.reshape(B, N_sb, K, Nt, D).permute(0, 2, 3, 1, 4).contiguous()
        if as_active:
            # [B,N_sb,K,D] -> [B,K,1,N_sb,D]; no extra attention or projection.
            # Do not branch on alpha: its derivative must exist at alpha=0.
            W_out = W_out + torch.tanh(self.alpha) * msg.permute(0, 2, 1, 3).unsqueeze(2)
        return W_out


class AdaptiveHybridPrecoderASDirect(AdaptiveHybridPrecoder):
    """Full UE architecture with a gated, user-specific AS residual at the BS.

    Construct before creating an optimizer. Configuration maxima remain the
    architecture maxima, independently of the supported allocation grid. A
    pure-AS specialist still needs full-width FDMA modules and a singleton
    running_pwr indexed by cfg_idx=0. No trainer/default factory is changed.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Transfer existing initialized child modules, preserving keys, values,
        # independent layer parameters, and RNG state. Only alpha is new.
        self.layers = nn.ModuleList(_ASDirectRefinement(layer) for layer in self.layers)

    @property
    def architecture(self):
        return "as_direct_v1"

    @property
    def gate_state_keys(self):
        return tuple(f"layers.{index}.alpha" for index in range(len(self.layers)))

    @property
    def architecture_metadata(self):
        return {
            "architecture": self.architecture,
            "base_model": "AdaptiveHybridPrecoder",
            "D_MODEL": self.D,
            "refinement_layers": len(self.layers),
            "additional_parameters": len(self.layers),
            "gate": "tanh(alpha)",
            "gate_initial_value": 0.0,
            "activation_condition": "d_a > 0",
            "D_F_MAX_PER_USER": self.cfg.D_F_MAX_PER_USER,
            "D_A_MAX_SHARED": self.cfg.D_A_MAX_SHARED,
            "allocation_grid": [list(point) for point in self.cfg.GAMMA_GRID],
        }

    def forward(
        self,
        H_dl,
        H_ul,
        snr_db,
        d_f,
        d_a,
        cfg_idx,
        H_ul_est=None,
    ):
        """Original physical/deployment path, with explicit AS activity only."""
        assert d_f % self.N_sb == 0 and d_a % self.N_sb == 0
        d_f_sb = d_f // self.N_sb
        d_a_sb = d_a // self.N_sb

        B = H_dl.shape[0]
        device = H_dl.device

        if H_ul_est is None:
            H_ul_est = H_ul

        if H_ul_est.shape != H_ul.shape:
            raise ValueError(
                "H_ul_est must have the same shape as H_ul: "
                f"H_ul={tuple(H_ul.shape)}, "
                f"H_ul_est={tuple(H_ul_est.shape)}."
            )

        if H_ul_est.device != H_ul.device:
            raise ValueError(
                "H_ul_est and H_ul must be on the same device."
            )

        if H_ul_est.dtype != H_ul.dtype:
            raise TypeError(
                "H_ul_est and H_ul must have the same dtype."
            )

        cond = self.cond_embed(d_f, d_a, snr_db, device)
        cond_5d = cond.view(1, 1, 1, 1, -1)
        cond_4d = cond.view(1, 1, 1, -1)

        H_dl_flat = H_dl.reshape(B * self.K, self.Nt, self.Nsc)
        z_f_flat, z_a_flat = self.encoder(H_dl_flat, d_f_sb, d_a_sb, cond)

        n_sym = z_f_flat.shape[1] + z_a_flat.shape[1]
        energy = (z_f_flat.abs() ** 2).sum() + (z_a_flat.abs() ** 2).sum()
        pwr_batch = energy / (B * self.K * n_sym) + 1e-9

        if self.training:
            with torch.no_grad():
                self.running_pwr[cfg_idx] = (
                    (1 - self.pwr_momentum) * self.running_pwr[cfg_idx]
                    + self.pwr_momentum * pwr_batch.detach()
                )
            scale = torch.sqrt(pwr_batch)
        else:
            scale = torch.sqrt(self.running_pwr[cfg_idx] + 1e-9)

        Z_f = (z_f_flat / scale).reshape(B, self.K, -1)
        Z_a = (z_a_flat / scale).reshape(B, self.K, -1)

        Y_f, Y_ac = self.transmit(Z_f, Z_a, H_ul, snr_db, d_f, d_a)

        if d_f > 0:
            Yf_dft = self.dft(Y_f)
            Yf_sb = Yf_dft.reshape(B, self.K, self.Nt, self.N_sb, d_f_sb)
            Yf_sb = self._pad_chunks(Yf_sb, self.pad_f, d_f_sb, self.d_f_sb_max)
            Yf_ri = torch.cat([Yf_sb.real, Yf_sb.imag], dim=-1)
            Y_f_feat = self.fdma_emb(Yf_ri)
        else:
            Y_f_feat = torch.zeros(
                B, self.K, self.Nt, self.N_sb, self.D, device=device
            )

        Y_f_feat = Y_f_feat + self.pos_emb.unsqueeze(1).unsqueeze(3)
        Y_f_feat = Y_f_feat + self.user_emb.unsqueeze(0).unsqueeze(3)
        Y_f_feat = Y_f_feat + self.subband_emb.unsqueeze(0).unsqueeze(0)
        Y_f_feat = Y_f_feat + cond_5d

        if d_a > 0:
            Yac_dft = self.dft(Y_ac)
            Yac_sb = Yac_dft.reshape(B, self.Nt, self.N_sb, d_a_sb)
            Yac_sb = self._pad_chunks(Yac_sb, self.pad_a, d_a_sb, self.d_a_sb_max)
            Yac_ri = torch.cat([Yac_sb.real, Yac_sb.imag], dim=-1)
            Y_ac_feat = self.aircomp_emb(Yac_ri)
            Y_ac_feat = Y_ac_feat + self.pos_emb[:, :, None, :]
            Y_ac_feat = Y_ac_feat + self.subband_emb[:, None, :, :]
            Y_ac_feat = Y_ac_feat + cond_4d
        else:
            Y_ac_feat = Y_f_feat.mean(dim=1).contiguous()

        x = Y_ac_feat.reshape(B * self.Nt, self.N_sb, self.D)
        Y_ac_feat = self.cross_subband_attn(x).reshape(
            B, self.Nt, self.N_sb, self.D
        )

        H_ul_est_dft = self.dft(H_ul_est)
        H_ul_r = H_ul_est_dft.real.reshape(
            B, self.K, self.Nt, self.N_sb, self.ul_per_sb
        )
        H_ul_i = H_ul_est_dft.imag.reshape(
            B, self.K, self.Nt, self.N_sb, self.ul_per_sb
        )
        H_ul_feat = self.ul_emb(
            torch.cat([H_ul_r, H_ul_i], dim=-1)
        )
        H_ul_feat = H_ul_feat + cond_5d

        W_feat = self.w_init(torch.cat([Y_f_feat, H_ul_feat], dim=-1))
        for layer in self.layers:
            W_feat = layer(W_feat, Y_f_feat, Y_ac_feat, as_active=(d_a > 0))

        h_raw = self.h_estimator(W_feat)
        h_split = h_raw.reshape(B, self.K, self.Nt, self.N_sb, self.sb_size, 2)
        H_hat = torch.complex(h_split[..., 0], h_split[..., 1]) \
                     .reshape(B, self.K, self.Nt, self.Nsc)

        W_zf = self.zf_solver(H_hat)

        if not self.training:
            frob_eval = (W_zf.abs() ** 2).sum(dim=(1, 2), keepdim=True)
            W_eval = W_zf * torch.sqrt(
                self.cfg.TOTAL_POWER / (frob_eval + 1e-9)
            )
            return W_eval

        W_zf_dir = W_zf / (torch.norm(W_zf, dim=(1, 2), keepdim=True) + 1e-8)

        res_raw = self.residual_head(W_feat)
        res_split = res_raw.reshape(B, self.K, self.Nt, self.N_sb, self.sb_size, 2)
        W_residual = torch.complex(res_split[..., 0], res_split[..., 1]) \
                          .reshape(B, self.K, self.Nt, self.Nsc) \
                          .permute(0, 2, 1, 3)

        W_train = W_zf_dir + self.residual_scale * W_residual
        frob_train = (W_train.abs() ** 2).sum(dim=(1, 2), keepdim=True)
        W_train = W_train * torch.sqrt(
            self.cfg.TOTAL_POWER / (frob_train + 1e-9)
        )

        return W_train, H_hat


def migrate_original_state_dict(model, original_state):
    """Explicit Full -> ASDirect weights-only initialization, never a resume.

    All source keys, shapes, and dtypes are checked before loading anything.
    Exactly the gate keys may be absent; only those keys are filled with zero.
    No slicing, dtype conversion, prefix filtering, or partial loading is used.
    For an ASDirect checkpoint use model.load_state_dict(state, strict=True)
    directly instead; a missing gate must remain an error on normal restore.
    """
    if not isinstance(model, AdaptiveHybridPrecoderASDirect):
        raise TypeError("Migration requires AdaptiveHybridPrecoderASDirect.")
    if not isinstance(original_state, Mapping):
        raise TypeError("original_state must be the original tensor state_dict.")

    target = model.state_dict()
    gate_keys = set(model.gate_state_keys)
    expected = set(target) - gate_keys
    actual = set(original_state)
    if actual != expected:
        raise ValueError(
            "Original state_dict key mismatch: "
            f"missing={sorted(expected - actual, key=str)}, "
            f"unexpected={sorted(actual - expected, key=str)}"
        )
    for name in sorted(expected):
        source = original_state[name]
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"{name}: expected a tensor, got {type(source).__name__}.")
        if source.shape != target[name].shape:
            raise ValueError(
                f"{name}: shape mismatch {tuple(source.shape)} != {tuple(target[name].shape)}"
            )
        if source.dtype != target[name].dtype:
            raise TypeError(f"{name}: dtype mismatch {source.dtype} != {target[name].dtype}")

    migrated = OrderedDict(original_state.items())
    if hasattr(original_state, "_metadata"):
        migrated._metadata = deepcopy(original_state._metadata)
    for name in model.gate_state_keys:
        migrated[name] = torch.zeros_like(target[name])
    return model.load_state_dict(migrated, strict=True)
