"""BS-only AS receiver residual over the unchanged Full precoder.

The only Full-forward insertion is after Stage-1 w_init. The local forward is
necessary because the protected original exposes no intermediate injection API;
there are no hooks, caches, or monkey patches. Tests compare its AST to Full
after removing exactly that insertion. Old encoder, link, context, refinement,
representation and deployed RZF operations remain untouched.
"""

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import math

import torch
from torch import nn

from .adaptive_hybrid import AdaptiveHybridPrecoder


# Explicit whitelist: no old key may be omitted during Full -> receiver migration.
AS_RECEIVER_STATE_KEYS = (
    "as_receiver.position_embedding",
    "as_receiver.token_proj.weight",
    "as_receiver.token_proj.bias",
    "as_receiver.query_proj.weight",
    "as_receiver.query_proj.bias",
    "as_receiver.attention.in_proj_weight",
    "as_receiver.attention.in_proj_bias",
    "as_receiver.attention.out_proj.weight",
    "as_receiver.attention.out_proj.bias",
    "as_receiver.output_proj.weight",
    "as_receiver.output_proj.bias",
)


def _receiver_settings(cfg):
    settings = getattr(cfg, "receiver_settings", {})
    if not isinstance(settings, Mapping):
        raise TypeError("cfg.receiver_settings must be a mapping when provided.")
    floor = getattr(cfg, "AS_RECEIVER_LAMBDA_FLOOR", settings.get("lambda_floor", 1e-6))
    dtype = getattr(cfg, "AS_RECEIVER_SOLVE_DTYPE", settings.get("solve_dtype", "complex64"))
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor <= 0:
        raise ValueError("AS receiver lambda_floor must be a positive finite number.")
    if dtype != "complex64":
        raise ValueError("as_receiver_v1 fixes regularized solve precision to complex64.")
    return float(floor), dtype


def regularized_recovery(
    Y_ac, H_ul_est, snr_db, d_f, d_a, lambda_floor=1e-6, solve_dtype="complex64"
):
    """Recover active user codewords using BS-available received data only.

    G[b,t] = H_ul_est[b,:,:,K*d_f+t].T, with no physical-channel oracle.
    The noise moment averages all B/Nt/active-Da entries, matching the scope of
    the existing relative received-power noise convention. This is an estimate
    from noisy Y_ac, NOT the simulator's noiseless signal or true noise power.
    Returns differentiable estimates; no signal, power, or state is cached.
    """
    if solve_dtype != "complex64":
        raise ValueError("as_receiver_v1 only supports complex64 recovery.")
    if isinstance(lambda_floor, bool) or not isinstance(lambda_floor, (int, float)) or not math.isfinite(lambda_floor) or lambda_floor <= 0:
        raise ValueError("lambda_floor must be positive and finite.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (d_f, d_a)):
        raise TypeError("d_f and d_a must be integers.")
    if d_f < 0 or d_a <= 0:
        raise ValueError("Recovery requires d_f >= 0 and active d_a > 0.")
    if Y_ac.ndim != 3 or H_ul_est.ndim != 4:
        raise ValueError("Expected Y_ac[B,Nt,Da] and H_ul_est[B,K,Nt,Nsc].")
    batch, users, antennas, subcarriers = H_ul_est.shape
    if tuple(Y_ac.shape) != (batch, antennas, d_a):
        raise ValueError("Y_ac must match H_ul_est batch/antennas and the active d_a.")
    if not Y_ac.is_complex() or not H_ul_est.is_complex():
        raise TypeError("AS recovery requires complex received data and BS channel estimates.")
    if Y_ac.device != H_ul_est.device:
        raise ValueError("Y_ac and H_ul_est must use the same device.")
    start = users * d_f
    if start + d_a > subcarriers:
        raise ValueError("The physical AS slice K*d_f:K*d_f+d_a exceeds the channel.")
    snr = float(snr_db.detach().item()) if torch.is_tensor(snr_db) else float(snr_db)
    if not math.isfinite(snr):
        raise ValueError("Feedback SNR must be finite.")
    gamma = 10.0 ** (snr / 10.0)
    # Explicit dtype is local to this branch; no global backend setting changes.
    received = Y_ac.to(torch.complex64)
    channel = H_ul_est[:, :, :, start:start + d_a].to(torch.complex64)
    g = channel.permute(0, 3, 2, 1)                 # [B,Da,Nt,K]
    y = received.permute(0, 2, 1).unsqueeze(-1)     # [B,Da,Nt,1]
    noise_est = received.abs().square().mean() / (1.0 + gamma)
    regularization = noise_est.clamp_min(float(lambda_floor))
    g_h = g.conj().transpose(-2, -1)
    identity = torch.eye(users, dtype=torch.complex64, device=received.device)
    system = g_h @ g + regularization * identity
    z_hat = torch.linalg.solve(system, g_h @ y).squeeze(-1)
    inverse_action = torch.linalg.solve(system, identity.expand(batch, d_a, users, users))
    reliability = (
        1.0 - (regularization * inverse_action).diagonal(dim1=-2, dim2=-1).real
    ).clamp(0.0, 1.0)
    return {
        "z_hat": z_hat,                             # [B,Da,K]
        "reliability": reliability,                 # [B,Da,K]
        "noise_est": noise_est,
        "regularization": regularization,
    }


class ASReceiverV1(nn.Module):
    """Per-user/subband codeword attention queried by existing Stage-1 angles."""

    def __init__(self, cfg, *, lambda_floor=1e-6, solve_dtype="complex64"):
        super().__init__()
        self.d_model = cfg.D_MODEL
        self.num_subbands = cfg.NUM_SUBBANDS
        self.max_symbols_per_subband = cfg.D_A_MAX_SHARED // cfg.NUM_SUBBANDS
        self.lambda_floor = lambda_floor
        self.solve_dtype = solve_dtype
        self.token_proj = nn.Linear(3, 128)
        self.query_proj = nn.Linear(self.d_model, 128)
        self.position_embedding = nn.Parameter(
            torch.randn(self.num_subbands, self.max_symbols_per_subband, 128) * 0.02
        )
        self.attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
        self.output_proj = nn.Linear(128, self.d_model)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def recover(self, Y_ac, H_ul_est, snr_db, d_f, d_a):
        return regularized_recovery(
            Y_ac, H_ul_est, snr_db, d_f, d_a,
            lambda_floor=self.lambda_floor, solve_dtype=self.solve_dtype,
        )

    def _codeword_tokens(self, z_hat, reliability):
        batch, active, users = z_hat.shape
        if active % self.num_subbands:
            raise ValueError("Active AS symbols must divide into the existing subbands.")
        active_per_subband = active // self.num_subbands
        if not 0 < active_per_subband <= self.max_symbols_per_subband:
            raise ValueError("AS prefix exceeds the retained maximum codeword width.")
        features = torch.stack((z_hat.real, z_hat.imag, reliability), dim=-1)
        # Physical serialization already concatenates each subband's active prefix.
        features = features.permute(0, 2, 1, 3).reshape(
            batch, users, self.num_subbands, active_per_subband, 3
        )
        features = features.to(dtype=self.token_proj.weight.dtype)
        tokens = self.token_proj(features)
        position = self.position_embedding[:, :active_per_subband, :]
        return tokens + position.unsqueeze(0).unsqueeze(0)

    def forward(self, W_init, Y_ac, H_ul_est, snr_db, d_f, d_a):
        # The main model bypasses the entire module for Da=0. Also make a direct
        # module call safe without dereferencing nonexistent received AS symbols.
        if d_a == 0:
            return torch.zeros_like(W_init)
        batch, users, antennas, subbands, width = W_init.shape
        if subbands != self.num_subbands or width != self.d_model:
            raise ValueError("Stage-1 query shape is incompatible with the AS receiver.")
        recovery = self.recover(Y_ac, H_ul_est, snr_db, d_f, d_a)
        if recovery["z_hat"].shape[0] != batch or recovery["z_hat"].shape[2] != users:
            raise ValueError("Received user/batch dimensions do not match Stage-1 queries.")
        tokens = self._codeword_tokens(recovery["z_hat"], recovery["reliability"])
        tokens = tokens.reshape(batch * users * subbands, -1, 128)
        queries = W_init.permute(0, 1, 3, 2, 4).reshape(
            batch * users * subbands, antennas, width
        )
        queries = self.query_proj(queries)
        attended, _ = self.attention(queries, tokens, tokens, need_weights=False)
        residual = self.output_proj(attended)
        return residual.reshape(batch, users, subbands, antennas, width).permute(
            0, 1, 3, 2, 4
        ).contiguous()


class AdaptiveHybridPrecoderASReceiver(AdaptiveHybridPrecoder):
    """Full architecture plus a zero-initialized, ungated Stage-1 AS receiver."""

    def __init__(self, cfg):
        floor, solve_dtype = _receiver_settings(cfg)
        if cfg.D_MODEL != 256:
            raise ValueError("as_receiver_v1 retains the Full D_MODEL=256 architecture.")
        super().__init__(cfg)
        # Construct only after the entire original model: same-seed old parameters
        # are unchanged. New initialization consumes RNG; callers must not infer
        # common training noise/dropout solely from the same initialization seed.
        self.as_receiver = ASReceiverV1(cfg, lambda_floor=floor, solve_dtype=solve_dtype)
        self.receiver_enabled = True  # Nonpersistent inference ablation switch.

    @property
    def architecture(self):
        return "as_receiver_v1"

    @property
    def added_state_keys(self):
        return AS_RECEIVER_STATE_KEYS

    @property
    def receiver_config_metadata(self):
        return {
            "width": 128, "heads": 4, "dropout": 0.0,
            "lambda_floor": self.as_receiver.lambda_floor,
            "solve_dtype": self.as_receiver.solve_dtype,
            "noise_estimator": "mean(abs(Y_ac)**2)/(1+10**(feedback_snr_db/10))",
            "noise_reduction": "all_batch_antennas_active_AS_symbols",
            "lambda_rule": "max(noise_est,lambda_floor)",
            "reliability": "clip(1-real(diag(lambda*solve(A,I))),0,1)",
            "physical_slice": "K*d_f:K*d_f+d_a",
            "token_features": ["real(z_hat)", "imag(z_hat)", "reliability"],
            "position_encoding": "learned_per_subband_prefix",
            "query_source": "Stage1_W_init_per_angle",
            "output_width": self.D,
            "output_initialization": "zero_weight_and_bias",
            "scalar_gate": False,
        }

    @property
    def architecture_metadata(self):
        return {
            "architecture": self.architecture,
            "model_class": type(self).__name__,
            "base_model": "AdaptiveHybridPrecoder",
            "ue_ffn_dim": 2048,
            "D_MODEL": self.D,
            "refinement_layers": len(self.layers),
            "D_F_MAX_PER_USER": self.cfg.D_F_MAX_PER_USER,
            "D_A_MAX_SHARED": self.cfg.D_A_MAX_SHARED,
            "allocation_grid": [list(point) for point in self.cfg.GAMMA_GRID],
            "receiver": self.receiver_config_metadata,
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
        """
        d_f: 当前每用户 FDMA 维度 (int), d_a: 当前共享 AirComp 维度 (int)
        cfg_idx: 该 (d_f, d_a) 在 cfg.GAMMA_GRID 中的索引 (用于 running power)
        约束: 整 batch 共享同一 (d_f, d_a) — 物理上 AirComp 要求小区级同步 γ
        """
        assert d_f % self.N_sb == 0 and d_a % self.N_sb == 0
        d_f_sb = d_f // self.N_sb
        d_a_sb = d_a // self.N_sb

        # B = H_dl.shape[0]
        # device = H_dl.device

        # cond = self.cond_embed(d_f, d_a, snr_db, device)
        B = H_dl.shape[0]
        device = H_dl.device

        # --------------------------------------------------------------
        # Feedback-link CSI available at the BS.
        #
        # H_ul:
        #     True physical feedback channel. It is always used by the
        #     actual FDMA/shared-spectrum transmission.
        #
        # H_ul_est:
        #     BS-side estimate of the feedback channel. It is used only
        #     as receiver-side CSI / decoder input.
        #
        # Defaulting to H_ul exactly reproduces the original perfect-CSI
        # behavior and keeps all existing training/evaluation code
        # backward compatible.
        # --------------------------------------------------------------
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

        # ---------- 1. UE Encoder ----------
        H_dl_flat = H_dl.reshape(B * self.K, self.Nt, self.Nsc)
        z_f_flat, z_a_flat = self.encoder(H_dl_flat, d_f_sb, d_a_sb, cond)

        # ---------- 2. 功率归一化 (per-config running stats) ----------
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

        # ---------- 3. 物理层传输 ----------
        Y_f, Y_ac = self.transmit(Z_f, Z_a, H_ul, snr_db, d_f, d_a)

        # ---------- 4. BS 端特征化 ----------
        # ----- FDMA -----
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

        # ----- AirComp -----
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
            # 公平补丁 (运行时分支): 纯 FDMA 时用跨用户均值合成全局上下文
            Y_ac_feat = Y_f_feat.mean(dim=1).contiguous()

        # [BUG 修复] cross-subband attention: 直接 reshape, 不再置乱
        x = Y_ac_feat.reshape(B * self.Nt, self.N_sb, self.D)
        Y_ac_feat = self.cross_subband_attn(x).reshape(
            B, self.Nt, self.N_sb, self.D
        )

        # ----- Uplink 先验 -----
        # H_ul_dft = self.dft(H_ul)
        # H_ul_r = H_ul_dft.real.reshape(B, self.K, self.Nt, self.N_sb, self.ul_per_sb)
        # H_ul_i = H_ul_dft.imag.reshape(B, self.K, self.Nt, self.N_sb, self.ul_per_sb)
        # H_ul_feat = self.ul_emb(torch.cat([H_ul_r, H_ul_i], dim=-1))
        # H_ul_feat = H_ul_feat + cond_5d


        # ----- BS-side feedback-link CSI -----
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

        # ---------- 5. W_feat 初始化 + L 层 Unfolded WMMSE ----------
        W_feat = self.w_init(torch.cat([Y_f_feat, H_ul_feat], dim=-1))
        if self.receiver_enabled and d_a > 0:
            W_feat = W_feat + self.as_receiver(
                W_feat, Y_ac, H_ul_est, snr_db, d_f, d_a
            )
        for layer in self.layers:
            W_feat = layer(W_feat, Y_f_feat, Y_ac_feat)

        # ---------- 6. ZF-Residual 双头 (与 V4 一致) ----------
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
    """Strict Full -> fresh ASReceiver initialization, never a resume shortcut.

    Validate every original key/shape/dtype before changing the model. Only the
    explicit new-key whitelist is filled from the freshly initialized receiver.
    Its output head must already be zero; this function never resets weights.
    Restore trained ASReceiver states with load_state_dict(..., strict=True).
    """
    if not isinstance(model, AdaptiveHybridPrecoderASReceiver):
        raise TypeError("Migration requires AdaptiveHybridPrecoderASReceiver.")
    if not isinstance(original_state, Mapping):
        raise TypeError("original_state must be the original tensor state_dict.")
    target = model.state_dict()
    added = set(model.added_state_keys)
    actual_added = {name for name in target if name.startswith("as_receiver.")}
    if actual_added != added:
        raise RuntimeError("AS receiver state differs from the explicit migration whitelist.")
    expected, actual = set(target) - added, set(original_state)
    if actual != expected:
        raise ValueError(
            "Original state_dict key mismatch: "
            f"missing={sorted(expected - actual, key=str)}, "
            f"unexpected={sorted(actual - expected, key=str)}"
        )
    for name in sorted(expected):
        value = original_state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}: expected a tensor, got {type(value).__name__}.")
        if value.shape != target[name].shape:
            raise ValueError(
                f"{name}: shape mismatch {tuple(value.shape)} != {tuple(target[name].shape)}"
            )
        if value.dtype != target[name].dtype:
            raise TypeError(f"{name}: dtype mismatch {value.dtype} != {target[name].dtype}")
    if any(torch.count_nonzero(parameter.detach()).item() != 0
           for parameter in model.as_receiver.output_proj.parameters()):
        raise ValueError("Original migration requires a fresh zero output projection; use strict restore for trained receivers.")
    migrated = OrderedDict((name, original_state[name] if name in expected else value)
                           for name, value in target.items())
    if hasattr(target, "_metadata"):
        migrated._metadata = deepcopy(target._metadata)
        if hasattr(original_state, "_metadata"):
            migrated._metadata.update(deepcopy(original_state._metadata))
    return model.load_state_dict(migrated, strict=True)
