# models/hybrid_precoder.py
"""
Hybrid Precoder Encoder with Decoupled FDMA + AirComp Branches
==============================================================
端到端 Subband-Aware Architecture (V3 终极版)
-----------------------------------------------------
1. UE 端解除全局 Pooling 封印，按 Subband 独立池化特征。
2. FDMA 和 AirComp 均输出具备真实频域起伏的子带码字。
"""
import torch
import torch.nn as nn
from utils.layers import DFTLayer


class HybridPrecodeEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim_fdma = cfg.DIM_FDMA_PER_USER
        self.dim_aircomp = cfg.DIM_AIRCOMP_SHARED
        self.Nt = cfg.ANTENNAS
        self.Nsc = cfg.SUBCARRIERS
        self.D = cfg.D_MODEL
        self.N_sb = getattr(cfg, 'NUM_SUBBANDS', 8)
        self.dft = DFTLayer(self.Nt, dim=-2)

        # ========================================================
        # FDMA 分支 (subband-aware)
        # ========================================================
        if self.dim_fdma > 0:
            assert self.dim_fdma % self.N_sb == 0, \
                f"dim_fdma={self.dim_fdma} 必须能被 N_sb={self.N_sb} 整除"
            self.dim_fdma_per_sb = self.dim_fdma // self.N_sb
        else:
            self.dim_fdma_per_sb = 0

        self.fdma_input_proj = nn.Linear(cfg.SUBCARRIERS * 2, self.D)
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

        if self.dim_fdma > 0:
            self.fc_fdma = nn.Linear(self.D, self.dim_fdma_per_sb * 2)
            self.fdma_sb_query = nn.Parameter(
                torch.randn(1, self.N_sb, self.D) * 0.02
            )
        else:
            self.fc_fdma = None
            self.fdma_sb_query = None

        # ========================================================
        # AirComp 分支 (subband-aware)
        # ========================================================
        if self.dim_aircomp > 0:
            assert self.dim_aircomp % self.N_sb == 0, \
                f"dim_aircomp={self.dim_aircomp} 必须能被 N_sb={self.N_sb} 整除"
            self.dim_ac_per_sb = self.dim_aircomp // self.N_sb

            self.aircomp_input_proj = nn.Linear(self.Nt * 2, self.D)
            self.aircomp_input_norm = nn.LayerNorm(self.D)

            aircomp_layer = nn.TransformerEncoderLayer(
                d_model=self.D, nhead=cfg.NUM_HEADS,
                batch_first=True, dropout=cfg.DROPOUT
            )
            self.aircomp_backbone = nn.TransformerEncoder(
                aircomp_layer, num_layers=cfg.NUM_ENCODER_LAYERS
            )

            self.sc_pos_emb = nn.Parameter(
                torch.randn(1, self.Nsc, self.D) * 0.02
            )

            self.aircomp_attn_pool = nn.Sequential(
                nn.Linear(self.D, self.D // 4), nn.GELU(),
                nn.Linear(self.D // 4, 1)
            )
            self.fc_aircomp = nn.Linear(self.D, self.dim_ac_per_sb * 2)
        else:
            self.aircomp_input_proj = None
            self.aircomp_backbone = None
            self.sc_pos_emb = None
            self.aircomp_attn_pool = None
            self.fc_aircomp = None
            self.dim_ac_per_sb = 0

    def _extract_eigenvector_per_sc(self, h_dl_flat):
        h_perm = h_dl_flat.permute(0, 2, 1)  # [BK, Nsc, Nt]
        h_norm = torch.norm(h_perm, dim=-1, keepdim=True) + 1e-9
        eigvec = h_perm / h_norm
        return eigvec

    def forward(self, h_dl_flat):
        BK = h_dl_flat.shape[0]
        device = h_dl_flat.device

        # ========================================================
        # FDMA 分支: subband-aware pooling
        # ========================================================
        h_ang = self.dft(h_dl_flat)
        x_fdma = torch.cat([h_ang.real, h_ang.imag], dim=-1)
        x_fdma = self.fdma_input_norm(self.fdma_input_proj(x_fdma))
        feat_fdma = self.fdma_backbone(x_fdma)  # [BK, Nt, D]

        if self.fc_fdma is not None:
            feat_for_sb = feat_fdma.unsqueeze(1)  # [BK, 1, Nt, D]
            sb_query = self.fdma_sb_query.unsqueeze(2)  # [1, N_sb, 1, D]
            feat_sb = feat_for_sb + sb_query  # [BK, N_sb, Nt, D]

            weights_f_logits = self.fdma_attn_pool(feat_sb)  # [BK, N_sb, Nt, 1]
            weights_f = torch.softmax(weights_f_logits, dim=2)
            feat_fdma_per_sb = torch.sum(feat_sb * weights_f, dim=2)  # [BK, N_sb, D]

            z_f_per_sb = self.fc_fdma(feat_fdma_per_sb)  # [BK, N_sb, dim_fdma_per_sb*2]
            z_f = z_f_per_sb.reshape(BK, self.dim_fdma * 2)
        else:
            z_f = torch.zeros(BK, 0, device=device)

        # ========================================================
        # AirComp 分支: subband-aware pooling
        # ========================================================
        if self.dim_aircomp > 0:
            eigvec = self._extract_eigenvector_per_sc(h_dl_flat)

            downsample_factor = 4
            eigvec = eigvec[:, ::downsample_factor, :]  # [BK, 128, Nt]

            x_ac = torch.cat([eigvec.real, eigvec.imag], dim=-1)
            x_ac = self.aircomp_input_norm(self.aircomp_input_proj(x_ac))
            x_ac = x_ac + self.sc_pos_emb[:, ::downsample_factor, :]

            feat_ac = self.aircomp_backbone(x_ac)  # [BK, 128, D]

            Nsc_down = feat_ac.shape[1]  # 128
            sb_size_enc = Nsc_down // self.N_sb  # 16
            
            # 将频域切分为 N_sb 份
            feat_ac_sb = feat_ac.reshape(BK, self.N_sb, sb_size_enc, self.D)

            weights_a_logits = self.aircomp_attn_pool(feat_ac_sb)
            weights_a = torch.softmax(weights_a_logits, dim=2)
            # 在每份内部池化，保留 Subband 独立特征
            feat_ac_per_sb = torch.sum(feat_ac_sb * weights_a, dim=2)  # [BK, N_sb, D]

            z_a_per_sb = self.fc_aircomp(feat_ac_per_sb)  # [BK, N_sb, dim_ac_per_sb*2]
            z_a = z_a_per_sb.reshape(BK, self.dim_aircomp * 2)
        else:
            z_a = torch.zeros(BK, 0, device=device)

        # ========================================================
        # 功率归一化 (batch-wise)
        # ========================================================
        if self.dim_fdma > 0 and self.dim_aircomp > 0:
            z_combined = torch.cat([z_f, z_a], dim=-1)
            pwr_batch = torch.mean(z_combined ** 2) + 1e-9
            scale = torch.sqrt(pwr_batch)
            z_f_norm = z_f / scale
            z_a_norm = z_a / scale
        elif self.dim_fdma > 0:
            pwr_batch = torch.mean(z_f ** 2) + 1e-9
            z_f_norm = z_f / torch.sqrt(pwr_batch)
            z_a_norm = z_a
        else:
            pwr_batch = torch.mean(z_a ** 2) + 1e-9
            z_a_norm = z_a / torch.sqrt(pwr_batch)
            z_f_norm = z_f

        # ========================================================
        # 转 complex 输出
        # ========================================================
        if self.dim_fdma > 0:
            z_f_complex = torch.complex(z_f_norm[:, :self.dim_fdma], z_f_norm[:, self.dim_fdma:])
        else:
            z_f_complex = torch.zeros(BK, 0, dtype=torch.cfloat, device=device)

        if self.dim_aircomp > 0:
            z_a_complex = torch.complex(z_a_norm[:, :self.dim_aircomp], z_a_norm[:, self.dim_aircomp:])
        else:
            z_a_complex = torch.zeros(BK, 0, dtype=torch.cfloat, device=device)

        return z_f_complex, z_a_complex