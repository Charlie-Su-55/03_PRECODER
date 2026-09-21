#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swin-based analog FDMA baseline with deploy-consistent RZF training."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.baseline_common import (
    angular_delay_transform,
    closed_form_rzf,
    fdma_feedback_mrc,
    inverse_angular_delay_transform,
)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H//window_size, W//window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Embedding dimension must divide num_heads.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, window_size**2, window_size**2)
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C//self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn + self.relative_bias.unsqueeze(0), dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        ws = self.window_size
        if H % ws != 0 or W % ws != 0:
            raise ValueError(f"Feature map {(H, W)} is not divisible by window size {ws}.")
        shortcut = x
        windows = window_partition(self.norm1(x).view(B, H, W, C), ws).view(-1, ws*ws, C)
        windows = self.attn(windows).view(-1, ws, ws, C)
        x = window_reverse(windows, ws, H, W).view(B, H*W, C)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W


class SwinCFNetEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbedding(2, cfg.SWIN_EMBED_DIM, cfg.SWIN_PATCH_SIZE)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                cfg.SWIN_EMBED_DIM, cfg.SWIN_NUM_HEADS,
                cfg.SWIN_WINDOW_SIZE, cfg.SWIN_MLP_RATIO,
            ) for _ in range(cfg.SWIN_NUM_BLOCKS)
        ])
        self.norm = nn.LayerNorm(cfg.SWIN_EMBED_DIM)
        self.fc = nn.Linear(cfg.SWIN_EMBED_DIM, cfg.SWIN_CODEWORD_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, H, W = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        return self.fc(self.norm(x).mean(dim=1))


class SwinCFNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.SWIN_NC // cfg.SWIN_PATCH_SIZE
        self.W = cfg.ANTENNAS // cfg.SWIN_PATCH_SIZE
        if self.H % cfg.SWIN_WINDOW_SIZE != 0 or self.W % cfg.SWIN_WINDOW_SIZE != 0:
            raise ValueError("Decoder feature map is incompatible with window size.")
        self.fc = nn.Linear(
            cfg.SWIN_CODEWORD_DIM,
            self.H * self.W * cfg.SWIN_EMBED_DIM,
        )
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                cfg.SWIN_EMBED_DIM, cfg.SWIN_NUM_HEADS,
                cfg.SWIN_WINDOW_SIZE, cfg.SWIN_MLP_RATIO,
            ) for _ in range(cfg.SWIN_NUM_BLOCKS)
        ])
        self.norm = nn.LayerNorm(cfg.SWIN_EMBED_DIM)
        self.final_proj = nn.Linear(
            cfg.SWIN_EMBED_DIM, 2 * cfg.SWIN_PATCH_SIZE**2
        )

    def forward(self, codeword: torch.Tensor) -> torch.Tensor:
        B = codeword.shape[0]
        x = self.fc(codeword).view(B, self.H*self.W, self.cfg.SWIN_EMBED_DIM)
        for block in self.blocks:
            x = block(x, self.H, self.W)
        x = self.final_proj(self.norm(x))
        p = self.cfg.SWIN_PATCH_SIZE
        x = x.view(B, self.H, self.W, 2, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, 2, self.cfg.SWIN_NC, self.cfg.ANTENNAS)


class Swin_E2E_Precoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K_USERS
        self.Nt = cfg.ANTENNAS
        self.Nsc = cfg.SUBCARRIERS
        self.Nc = cfg.SWIN_NC
        self.D_f = cfg.D_F
        self.shared = bool(cfg.SHARE_USER_WEIGHTS)

        if cfg.SWIN_CODEWORD_DIM != 2 * self.D_f:
            raise ValueError("SWIN_CODEWORD_DIM must equal 2*D_f.")
        if self.K * self.D_f > self.Nsc:
            raise ValueError(
                f"FDMA uses {self.K*self.D_f} subcarriers, but Nsc={self.Nsc}."
            )

        if self.shared:
            self.encoder = SwinCFNetEncoder(cfg)
            self.decoder = SwinCFNetDecoder(cfg)
        else:
            self.encoders = nn.ModuleList([SwinCFNetEncoder(cfg) for _ in range(self.K)])
            self.decoders = nn.ModuleList([SwinCFNetDecoder(cfg) for _ in range(self.K)])

    def _encode(self, H_ad: torch.Tensor) -> torch.Tensor:
        B, K = H_ad.shape[:2]
        if self.shared:
            codewords = self.encoder(H_ad.reshape(B*K, 2, self.Nc, self.Nt))
            return codewords.view(B, K, -1)
        return torch.stack([self.encoders[k](H_ad[:, k]) for k in range(K)], dim=1)

    def _decode(self, received: torch.Tensor) -> torch.Tensor:
        B, K = received.shape[:2]
        if self.shared:
            H_ad = self.decoder(received.reshape(B*K, -1))
            return H_ad.view(B, K, 2, self.Nc, self.Nt)
        return torch.stack([self.decoders[k](received[:, k]) for k in range(K)], dim=1)

    def forward(self, H_dl: torch.Tensor, H_ul: torch.Tensor, snr_db: torch.Tensor | float = 10.0, H_ul_est: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        H_ad = angular_delay_transform(H_dl, self.Nc)
        codewords = self._encode(H_ad)
        received = fdma_feedback_mrc(codewords, H_ul, snr_db, self.D_f, H_ul_est=H_ul_est)
        H_ad_recon = self._decode(received)
        H_surrogate = inverse_angular_delay_transform(H_ad_recon, self.Nsc)
        W = closed_form_rzf(
            H_surrogate,
            regularization=self.cfg.RZF_REG,
            total_power=self.cfg.TOTAL_POWER,
        )
        return {
            "precoder": W,
            "channel_surrogate": H_surrogate,
        }


if __name__ == "__main__":
    from configs.baseline_config import make_config

    cfg = make_config("swin", "k4_n128_d128", output_root="runs_baselines_smoke")
    model = Swin_E2E_Precoder(cfg)
    H_dl = torch.randn(2, cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS, dtype=torch.complex64)
    H_ul = torch.randn_like(H_dl)
    out = model(H_dl, H_ul, 15.0)
    print(out["precoder"].shape, out["channel_surrogate"].shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))