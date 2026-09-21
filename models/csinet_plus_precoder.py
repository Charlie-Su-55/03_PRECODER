#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CsiNet+-based analog FDMA baseline with deploy-consistent RZF training."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.baseline_common import (
    angular_delay_transform,
    closed_form_rzf,
    fdma_feedback_mrc,
    inverse_angular_delay_transform,
)


class RefineBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class CsiNetPlusEncoder(nn.Module):
    def __init__(self, cfg, codeword_dim_real: int):
        super().__init__()
        layers = []
        in_channels = 2
        for out_channels in cfg.CSINET_ENC_CHANNELS:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        layers.extend([
            RefineBlock(cfg.CSINET_ENC_CHANNELS[-1]),
            RefineBlock(cfg.CSINET_ENC_CHANNELS[-1]),
        ])
        self.conv_layers = nn.Sequential(*layers)
        flat_dim = cfg.CSINET_ENC_CHANNELS[-1] * cfg.CSINET_NC * cfg.ANTENNAS
        self.fc = nn.Linear(flat_dim, codeword_dim_real)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv_layers(x).flatten(1))


class CsiNetPlusDecoder(nn.Module):
    def __init__(self, cfg, codeword_dim_real: int):
        super().__init__()
        self.nc = cfg.CSINET_NC
        self.nt = cfg.ANTENNAS
        self.first_channels = cfg.CSINET_DEC_CHANNELS[0]

        self.fc = nn.Linear(
            codeword_dim_real,
            self.first_channels * self.nc * self.nt,
        )
        layers = []
        in_channels = self.first_channels
        for out_channels in cfg.CSINET_DEC_CHANNELS[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        layers.extend([
            RefineBlock(cfg.CSINET_DEC_CHANNELS[-1]),
            RefineBlock(cfg.CSINET_DEC_CHANNELS[-1]),
        ])
        self.deconv_layers = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(
            cfg.CSINET_DEC_CHANNELS[-1], 2, 3, padding=1
        )

    def forward(self, codeword: torch.Tensor) -> torch.Tensor:
        B = codeword.shape[0]
        x = self.fc(codeword).view(B, self.first_channels, self.nc, self.nt)
        return self.final_conv(self.deconv_layers(x))


class CsiNetPlus_E2E_Precoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K_USERS
        self.Nt = cfg.ANTENNAS
        self.Nsc = cfg.SUBCARRIERS
        self.Nc = cfg.CSINET_NC
        self.D_f = cfg.D_F
        self.codeword_dim_real = 2 * self.D_f
        self.shared = bool(cfg.SHARE_USER_WEIGHTS)

        if self.K * self.D_f > self.Nsc:
            raise ValueError(
                f"FDMA uses {self.K*self.D_f} subcarriers, but Nsc={self.Nsc}."
            )

        if self.shared:
            self.encoder = CsiNetPlusEncoder(cfg, self.codeword_dim_real)
            self.decoder = CsiNetPlusDecoder(cfg, self.codeword_dim_real)
        else:
            self.encoders = nn.ModuleList([
                CsiNetPlusEncoder(cfg, self.codeword_dim_real)
                for _ in range(self.K)
            ])
            self.decoders = nn.ModuleList([
                CsiNetPlusDecoder(cfg, self.codeword_dim_real)
                for _ in range(self.K)
            ])

    def _encode(self, H_ad: torch.Tensor) -> torch.Tensor:
        B, K = H_ad.shape[:2]
        if self.shared:
            codewords = self.encoder(H_ad.reshape(B*K, 2, self.Nc, self.Nt))
            return codewords.view(B, K, -1)
        return torch.stack([
            self.encoders[k](H_ad[:, k]) for k in range(K)
        ], dim=1)

    def _decode(self, received: torch.Tensor) -> torch.Tensor:
        B, K = received.shape[:2]
        if self.shared:
            H_ad = self.decoder(received.reshape(B*K, -1))
            return H_ad.view(B, K, 2, self.Nc, self.Nt)
        return torch.stack([
            self.decoders[k](received[:, k]) for k in range(K)
        ], dim=1)

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

    cfg = make_config("csinet", "k4_n128_d128", output_root="runs_baselines_smoke")
    model = CsiNetPlus_E2E_Precoder(cfg)
    H_dl = torch.randn(2, cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS, dtype=torch.complex64)
    H_ul = torch.randn_like(H_dl)
    out = model(H_dl, H_ul, 15.0)
    print(out["precoder"].shape, out["channel_surrogate"].shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))