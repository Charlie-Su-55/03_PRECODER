# models/adaptive_hybrid.py
# -*- coding: utf-8 -*-
"""
Adaptive Hybrid FDMA-AirComp Precoder  (单 checkpoint γ 自适应版, 方案丙 Stage A)
==============================================================================
相对 V4 (hybrid_precoder.py + unfolded_wmmse_precoder.py) 的核心改动:

1. [合并]   Encoder 与 BS Decoder 合并到同一文件, 单一 nn.Module 入口。
2. [嵌套码字] Encoder 输出头固定为最大维度 (d_f_sb_max / d_a_sb_max),
            发送时按当前 (D_f, D_a) 做"前缀截断" → 随机 γ 训练自然诱导有序表示。
3. [可学习 Padding] BS 端接收符号 pad 到最大维度 (JSBANet 式可学习 ψ 向量),
            fdma_emb / aircomp_emb 线性层固定最大尺寸, 全 γ 共享权重。
4. [条件嵌入] cond = MLP([D_f/D_f_max, D_a/D_a_max, SNR/25]),
            加性注入 encoder 输入与 BS 三路特征流 (FiLM-lite, 不动标准件)。
5. [P2 修复] 功率归一化改为 per-config running statistics:
            训练用 batch 统计 (BN 式), 推理用冻结的 running_pwr[cfg_idx],
            消除 batch 统计泄漏导致的评估漂移。
6. [BUG 修复] 修复 V4 中 cross-subband attention 的 permute/reshape 置乱问题
            (原代码 [B,Nt,D,N_sb] reshape 成 (B*Nt, N_sb, D) 导致 token 错位)。

不变的部分: GNN_AggregationModule / UnfoldedWMMSELayer / zf_solver /
            ZF-residual 双头 / train-eval 输出协议, 与 V4 完全一致。

物理约定 (与 V4 保持一致以便公平对比, 但需注意):
- add_noise 的 sigma 按"该流自身接收功率"相对定义 → 两条流各自精确获得
  snr_db。这意味着 encoder 学到的 FDMA/AirComp 功率分配对噪声鲁棒性不起
  作用 (TODO: 物理上应改为绝对噪声底, 改动会影响与已有 baseline 的可比性,
  待与导师讨论后统一切换)。
"""
import torch
import torch.nn as nn
from utils.layers import DFTLayer


# ============================================================================
# 条件嵌入: (D_f, D_a, SNR) → [D] 向量
# ============================================================================
class ConditionEmbed(nn.Module):
    def __init__(self, d_model, d_f_max, d_a_max):
        super().__init__()
        self.d_f_max = float(max(d_f_max, 1))
        self.d_a_max = float(max(d_a_max, 1))
        self.net = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, d_f, d_a, snr_db, device):
        s = snr_db.item() if torch.is_tensor(snr_db) else float(snr_db)
        c = torch.tensor(
            [d_f / self.d_f_max, d_a / self.d_a_max, s / 25.0],
            device=device, dtype=torch.float32
        )
        return self.net(c)  # [D]


# ============================================================================
# Encoder (UE 端, 双分支, 最大维度输出 + 前缀截断)
# ============================================================================
class AdaptiveHybridEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Nt = cfg.ANTENNAS
        self.Nsc = cfg.SUBCARRIERS
        self.D = cfg.D_MODEL
        self.N_sb = cfg.NUM_SUBBANDS
        self.d_f_sb_max = cfg.D_F_MAX_PER_USER // self.N_sb
        self.d_a_sb_max = cfg.D_A_MAX_SHARED // self.N_sb
        self.ds_factor = 4

        self.dft = DFTLayer(self.Nt, dim=-2)

        # ---------------- FDMA 分支 (角域) ----------------
        self.fdma_input_proj = nn.Linear(self.Nsc * 2, self.D)
        self.fdma_input_norm = nn.LayerNorm(self.D)
        fdma_layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=cfg.NUM_HEADS,
            batch_first=True, dropout=cfg.DROPOUT
        )
        self.fdma_backbone = nn.TransformerEncoder(
            fdma_layer, num_layers=cfg.NUM_ENCODER_LAYERS
        )
        self.fdma_attn_pool = nn.Sequential(
            nn.Linear(self.D, self.D // 4), nn.GELU(),
            nn.Linear(self.D // 4, 1)
        )
        self.fdma_sb_query = nn.Parameter(
            torch.randn(1, self.N_sb, self.D) * 0.02
        )
        # 固定最大输出宽度, 前缀截断
        self.fc_fdma = nn.Linear(self.D, self.d_f_sb_max * 2)

        # ---------------- AirComp 分支 (归一化方向) ----------------
        self.aircomp_input_proj = nn.Linear(self.Nt * 2, self.D)
        self.aircomp_input_norm = nn.LayerNorm(self.D)
        ac_layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=cfg.NUM_HEADS,
            batch_first=True, dropout=cfg.DROPOUT
        )
        self.aircomp_backbone = nn.TransformerEncoder(
            ac_layer, num_layers=cfg.NUM_ENCODER_LAYERS
        )
        self.sc_pos_emb = nn.Parameter(
            torch.randn(1, self.Nsc, self.D) * 0.02
        )
        self.aircomp_attn_pool = nn.Sequential(
            nn.Linear(self.D, self.D // 4), nn.GELU(),
            nn.Linear(self.D // 4, 1)
        )
        self.fc_aircomp = nn.Linear(self.D, self.d_a_sb_max * 2)

    def forward(self, h_dl_flat, d_f_sb, d_a_sb, cond):
        """
        h_dl_flat: [BK, Nt, Nsc] complex
        d_f_sb / d_a_sb: 当前激活的 per-subband 维度 (int, 可为 0)
        cond: [D] 条件嵌入
        """
        BK = h_dl_flat.shape[0]
        device = h_dl_flat.device
        cond_b = cond.view(1, 1, -1)

        # ---------------- FDMA ----------------
        if d_f_sb > 0:
            h_ang = self.dft(h_dl_flat)
            x = torch.cat([h_ang.real, h_ang.imag], dim=-1)
            x = self.fdma_input_norm(self.fdma_input_proj(x)) + cond_b
            feat = self.fdma_backbone(x)                                   # [BK, Nt, D]
            feat_sb = feat.unsqueeze(1) + self.fdma_sb_query.unsqueeze(2)  # [BK, N_sb, Nt, D]
            w = torch.softmax(self.fdma_attn_pool(feat_sb), dim=2)
            pooled = (feat_sb * w).sum(dim=2)                              # [BK, N_sb, D]
            full = self.fc_fdma(pooled)                                    # [BK, N_sb, 2*d_f_sb_max]
            z = torch.complex(full[..., :self.d_f_sb_max],
                              full[..., self.d_f_sb_max:])                 # [BK, N_sb, d_f_sb_max]
            z_f = z[..., :d_f_sb].reshape(BK, self.N_sb * d_f_sb)          # 前缀截断
        else:
            z_f = torch.zeros(BK, 0, dtype=torch.cfloat, device=device)

        # ---------------- AirComp ----------------
        if d_a_sb > 0:
            eig = h_dl_flat.permute(0, 2, 1)                               # [BK, Nsc, Nt]
            eig = eig / (torch.norm(eig, dim=-1, keepdim=True) + 1e-9)
            eig = eig[:, ::self.ds_factor, :]                              # [BK, Nsc/4, Nt]
            x = torch.cat([eig.real, eig.imag], dim=-1)
            x = self.aircomp_input_norm(self.aircomp_input_proj(x))
            x = x + self.sc_pos_emb[:, ::self.ds_factor, :] + cond_b
            feat = self.aircomp_backbone(x)                                # [BK, Nsc', D]
            n_tok = feat.shape[1]
            feat_sb = feat.reshape(BK, self.N_sb, n_tok // self.N_sb, self.D)
            w = torch.softmax(self.aircomp_attn_pool(feat_sb), dim=2)
            pooled = (feat_sb * w).sum(dim=2)                              # [BK, N_sb, D]
            full = self.fc_aircomp(pooled)                                 # [BK, N_sb, 2*d_a_sb_max]
            z = torch.complex(full[..., :self.d_a_sb_max],
                              full[..., self.d_a_sb_max:])
            z_a = z[..., :d_a_sb].reshape(BK, self.N_sb * d_a_sb)          # 前缀截断
        else:
            z_a = torch.zeros(BK, 0, dtype=torch.cfloat, device=device)

        return z_f, z_a


# ============================================================================
# GNN-style Aggregation (与 V4 完全一致)
# ============================================================================
class GNN_AggregationModule(nn.Module):
    def __init__(self, d_model, d_a=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.d_a = d_a
        self.msg_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.user_query_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.pair_agg = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_a)
        )

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
        return A_feat


# ============================================================================
# Unfolded WMMSE Layer (与 V4 完全一致)
# ============================================================================
class UnfoldedWMMSELayer(nn.Module):
    def __init__(self, cfg, d_model, d_a=64, n_heads_solver=8, n_heads_cross=4):
        super().__init__()
        self.K = cfg.K_USERS
        self.Nt = cfg.ANTENNAS
        self.N_sb = cfg.NUM_SUBBANDS
        self.d_model = d_model
        self.d_a = d_a
        self.n_heads = n_heads_solver

        self.uv_estimator = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.A_estimator = GNN_AggregationModule(d_model, d_a=d_a, n_heads=4)

        self.w_solver_attn = nn.MultiheadAttention(
            d_model, num_heads=self.n_heads, batch_first=True
        )
        self.A_to_bias = nn.Linear(d_a, self.n_heads)

        self.cross_user_attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads_cross, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, W_feat, Y_f_feat, Y_ac_feat):
        B, K, Nt, N_sb, D = W_feat.shape
        H = self.n_heads

        W_pool = W_feat.mean(dim=2)
        Yf_pool = Y_f_feat.mean(dim=2)
        uv_input = torch.cat([W_pool, Yf_pool], dim=-1)
        uv_feat = self.uv_estimator(uv_input)
        uv_feat = uv_feat.permute(0, 2, 1, 3).contiguous()

        Y_ac_perm = Y_ac_feat.permute(0, 2, 1, 3).contiguous()
        A_feat = self.A_estimator(Y_ac_perm, uv_feat)
        attn_bias = self.A_to_bias(A_feat)
        attn_bias = attn_bias.permute(0, 1, 4, 2, 3).contiguous()

        attn_bias_K = attn_bias.unsqueeze(2).expand(B, N_sb, K, H, Nt, Nt).contiguous()
        attn_bias_K = attn_bias_K.reshape(B * N_sb * K * H, Nt, Nt)

        W_perm = W_feat.permute(0, 3, 1, 2, 4).contiguous()
        W_flat = W_perm.reshape(B * N_sb * K, Nt, D)

        attn_out, _ = self.w_solver_attn(
            W_flat, W_flat, W_flat,
            attn_mask=attn_bias_K
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
        return W_out


# ============================================================================
# 主模型: 单 checkpoint γ 自适应 Hybrid Precoder
# ============================================================================
class AdaptiveHybridPrecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K_USERS
        self.Nt = cfg.ANTENNAS
        self.Nsc = cfg.SUBCARRIERS
        self.D = cfg.D_MODEL
        self.L = cfg.NUM_UNFOLD_LAYERS
        self.d_a_gnn = cfg.GNN_AGG_DIM
        self.N_sb = cfg.NUM_SUBBANDS

        assert self.Nsc % self.N_sb == 0
        self.sb_size = self.Nsc // self.N_sb
        self.ul_per_sb = self.Nsc // self.N_sb

        self.d_f_sb_max = cfg.D_F_MAX_PER_USER // self.N_sb
        self.d_a_sb_max = cfg.D_A_MAX_SHARED // self.N_sb
        self.n_cfgs = len(cfg.GAMMA_GRID)

        self.dft = DFTLayer(cfg.ANTENNAS, dim=-2)
        self.encoder = AdaptiveHybridEncoder(cfg)
        self.cond_embed = ConditionEmbed(
            self.D, cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED
        )

        # ---- BS 端嵌入层: 固定最大尺寸 + 可学习 padding (JSBANet 式 ψ) ----
        self.fdma_emb = nn.Linear(self.d_f_sb_max * 2, self.D)
        self.aircomp_emb = nn.Linear(self.d_a_sb_max * 2, self.D)
        self.ul_emb = nn.Linear(self.ul_per_sb * 2, self.D)
        self.pad_f = nn.Parameter(torch.zeros(self.d_f_sb_max, 2))
        self.pad_a = nn.Parameter(torch.zeros(self.d_a_sb_max, 2))

        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.ANTENNAS, self.D))
        self.user_emb = nn.Parameter(torch.randn(cfg.K_USERS, 1, self.D) * 0.02)
        self.subband_emb = nn.Parameter(torch.randn(1, self.N_sb, self.D) * 0.02)

        cross_sb_layer = nn.TransformerEncoderLayer(
            d_model=self.D, nhead=cfg.NUM_HEADS, batch_first=True, dropout=0.0
        )
        self.cross_subband_attn = nn.TransformerEncoder(cross_sb_layer, num_layers=1)

        self.w_init = nn.Sequential(
            nn.Linear(self.D * 2, self.D), nn.GELU(),
            nn.Linear(self.D, self.D)
        )

        self.layers = nn.ModuleList([
            UnfoldedWMMSELayer(cfg, d_model=self.D, d_a=self.d_a_gnn)
            for _ in range(self.L)
        ])

        # ---- ZF-Residual 双头 (与 V4 一致) ----
        self.h_estimator = nn.Sequential(
            nn.Linear(self.D, self.D // 2), nn.GELU(),
            nn.Linear(self.D // 2, self.sb_size * 2)
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.D, self.D // 2), nn.GELU(),
            nn.Linear(self.D // 2, self.sb_size * 2)
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

        # ---- [P2 修复] per-config running power 统计 ----
        self.register_buffer('running_pwr', torch.ones(self.n_cfgs))
        self.pwr_momentum = 0.02

    # ------------------------------------------------------------------
    def add_noise(self, y, snr_db):
        # 注意: sigma 相对该流自身平均接收功率定义 (沿用 V4 约定, 见文件头 TODO)
        sigma = torch.sqrt(torch.mean(torch.abs(y) ** 2) * (10 ** (-snr_db / 10)) / 2)
        return y + (torch.randn_like(y) + 1j * torch.randn_like(y)) * sigma

    # ------------------------------------------------------------------
    def transmit(self, Z_f, Z_a, H_ul, snr_db, d_f, d_a):
        B = H_ul.shape[0]
        device = H_ul.device

        if d_f > 0:
            Y_f = torch.zeros(B, self.K, self.Nt, d_f,
                              dtype=torch.cfloat, device=device)
            for k in range(self.K):
                s = k * d_f
                Y_f[:, k] = H_ul[:, k, :, s:s + d_f] * Z_f[:, k].unsqueeze(1)
            Y_f = self.add_noise(Y_f, snr_db)
        else:
            Y_f = None

        if d_a > 0:
            start = self.K * d_f
            Y_ac = torch.sum(
                H_ul[:, :, :, start:start + d_a] * Z_a.unsqueeze(2),
                dim=1
            )
            Y_ac = self.add_noise(Y_ac, snr_db)
        else:
            Y_ac = None

        return Y_f, Y_ac

    # ------------------------------------------------------------------
    def _pad_chunks(self, Y_sb, pad_param, d_active, d_max):
        """ [..., d_active] complex → [..., d_max], 尾部填可学习 ψ """
        if d_active == d_max:
            return Y_sb
        pad_c = torch.complex(pad_param[:, 0], pad_param[:, 1])[d_active:]
        shape = list(Y_sb.shape[:-1]) + [d_max - d_active]
        pad = pad_c.view(*([1] * (Y_sb.dim() - 1)), -1).expand(shape)
        return torch.cat([Y_sb, pad], dim=-1)

    # ------------------------------------------------------------------
    def zf_solver(self, H_hat):
        B, K, Nt, Nsc = H_hat.shape
        device = H_hat.device

        H_per_sc = H_hat.permute(0, 3, 1, 2)
        H_eff = H_per_sc.conj()

        norm_factor = torch.norm(H_eff, dim=(2, 3), keepdim=True) + 1e-8
        H_eff_norm = H_eff / norm_factor

        H_eff_H = H_eff_norm.conj().transpose(-2, -1)
        H_gram = H_eff_norm @ H_eff_H

        reg = 1e-3 * torch.eye(K, device=device, dtype=torch.cfloat)
        H_gram_reg = H_gram + reg.unsqueeze(0).unsqueeze(0)

        W_zf_per_sc = H_eff_H @ torch.linalg.inv(H_gram_reg)
        return W_zf_per_sc.permute(0, 2, 3, 1)

    # ------------------------------------------------------------------
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