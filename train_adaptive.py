#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIN_ADAPTIVE: 单 checkpoint γ 自适应 Hybrid Precoder 训练
===========================================================
方案丙 Stage A: 固定 K=4, D_tot=128, 一个模型覆盖整个 (D_f, D_a) 网格。

与 03_precoder 训练脚本的区别:
1. 每个 step 从 GAMMA_GRID 随机采样一个 (D_f, D_a), 整 batch 共享 (小区级同步)。
2. [P0 修复] 全部配置使用同一套损失配方 (UNIFIED RECIPE), 不再 per-γ 手调。
3. 验证输出 5×4 的 (config × SNR) 矩阵, 直接观察 valley 是否在自适应模型下存活。
4. checkpoint 重命名为 adaptive_* 系列, 与已有 per-γ 模型完全隔离。
5. 固定随机种子 (M5/可复现性)。

运行: python train_adaptive.py
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

from models.adaptive_hybrid import AdaptiveHybridPrecoder
from utils import CDLChannelGenerator_Feedback, setup_gpu


# ============================================================================
# 配置 (自包含, 不依赖 configs/ 下的 per-γ 配置)
# ============================================================================
class AdaptiveConfig:
    # ----- 物理层 -----
    K_USERS = 4
    ANTENNAS = 32
    SUBCARRIERS = 128
    CARRIER_FREQ = 3.5e9
    SPEED = 1.0

    # ----- 资源网格 (与论文 Table II 的 K=4 sweep 完全一致) -----
    D_TOT = 128
    D_F_MAX_PER_USER = 32     # 覆盖 (32, 0) 纯 FDMA 极点
    D_A_MAX_SHARED = 128      # 覆盖 (0, 128) 纯 AirComp 极点
    NUM_SUBBANDS = 4          # 所有 D_f/D_a 取值都能被 4 整除 ✓
    GAMMA_GRID = [
        (32, 0),    # γ=0.00 纯 FDMA
        (24, 32),   # γ=0.25 局部峰
        (16, 64),   # γ=0.50 failure valley
        (8, 96),    # γ=0.75
        (0, 128),   # γ=1.00 纯 AirComp
    ]

    # ----- 模型 -----
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 4
    DROPOUT = 0.1
    NUM_UNFOLD_LAYERS = 5
    GNN_AGG_DIM = 48

    # ----- 训练总控 -----
    SEED = 42
    TOTAL_POWER = 1.0
    BATCH_SIZE = 48
    LR = 1e-4
    # 5 个配置混训, 等效每配置 ~6k steps; 比 per-γ 的 20k 少,
    # 靠跨配置的表示共享补偿。若收敛不足, 优先加到 40k。
    TOTAL_STEPS = 50000
    VAL_INTERVAL = 500
    VAL_BATCH = 64            # 建议有空时升到 256 重测 (小验证集是模型选择噪声源)
    WARMUP_PCT = 0.35
    GRAD_CLIP = 1.0

    # ----- 两阶段 + SNR 课程 (按 30k 总步数等比缩放) -----
    PHASE1_STEPS = 20000      # Phase1: 强 MSE/方向锚定; Phase2: 放开 Sum-Rate
    STAGE1_STEPS = 15000       # SNR 课程: 前 30% 高 SNR
    STAGE1_SNR = (20.0, 25.0)
    STAGE2_STEPS = 35000      # 中间 40%
    STAGE2_SNR = (10.0, 25.0)
    STAGE3_SNR = (0.0, 25.0)
    VAL_SNR_LIST = [0, 10, 20, 25]

    # ----- [P0] 统一损失配方: 所有 γ 共用, 严禁 per-γ 改动 -----
    # 沿用你跑通过的 "real-cos + 强MSE锚定" 配方;
    # 若要回到论文 III-C 的 abs 相位不变版本, 把 DIR_MODE 改成 'abs'
    DIR_MODE = 'real'         # 'real' (代码现状) | 'abs' (论文式29)
    PH1_W_DIR = 50.0
    PH1_W_MSE = 100.0
    PH2_W_DIR = 5.0
    PH2_W_MSE = 0.0

    # ----- 路径 (重命名, 与旧 ckp 隔离) -----
    EXP_NAME = (
        f"adaptive_uwmmse_{K_USERS}ue_{SUBCARRIERS}sc"
        f"_dtot{D_TOT}_sb{NUM_SUBBANDS}_unified_seed{SEED}"
    )
    SAVE_DIR = f"ckp/{EXP_NAME}"
    SAVE_PATH = f"{SAVE_DIR}/adaptive_latest.pth"
    BEST_PATH = f"{SAVE_DIR}/adaptive_best.pth"


cfg = AdaptiveConfig


# ============================================================================
# Sum Rate Loss (与 03_precoder 一致)
# ============================================================================
class SumRateLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, W, H_dl_true, noise_power):
        h_eff = torch.einsum('bkns,bnjs->bkjs', H_dl_true.conj(), W)
        power = h_eff.abs() ** 2
        sig_pwr = torch.stack(
            [power[:, i, i, :] for i in range(power.shape[1])], dim=1
        )
        int_pwr = power.sum(dim=2) - sig_pwr
        sinr = sig_pwr / (int_pwr + noise_power + self.eps)
        rate = torch.log2(1.0 + sinr)
        avg_rate = rate.sum(dim=1).mean()
        return -avg_rate, avg_rate.item()


# ============================================================================
# 辅助函数
# ============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_snr(step):
    if step <= cfg.STAGE1_STEPS:
        return np.random.uniform(*cfg.STAGE1_SNR)
    elif step <= cfg.STAGE2_STEPS:
        return np.random.uniform(*cfg.STAGE2_SNR)
    return np.random.uniform(*cfg.STAGE3_SNR)


def alignment_losses(H_hat, H_dl):
    """ 统一配方的方向损失 + MSE 损失 """
    H_hat_n = H_hat / (H_hat.abs().pow(2).sum(dim=2, keepdim=True).sqrt() + 1e-8)
    H_dl_n = H_dl / (H_dl.abs().pow(2).sum(dim=2, keepdim=True).sqrt() + 1e-8)
    inner = (H_hat_n * H_dl_n.conj()).sum(dim=2)
    if cfg.DIR_MODE == 'abs':
        cos_sim = inner.abs().mean()      # 论文式(29): 相位不变
    else:
        cos_sim = inner.real.mean()       # 代码现状: 强制相位对齐
    dir_loss = 1.0 - cos_sim
    mse_loss = torch.mean(torch.abs(H_hat - H_dl) ** 2)
    return dir_loss, mse_loss, cos_sim


@torch.no_grad()
def validate(model, H_dl_val, H_ul_val, criterion, device):
    """ 返回 {(cfg_idx, snr): rate} 的完整矩阵 """
    model.eval()
    table = {}
    for ci, (d_f, d_a) in enumerate(cfg.GAMMA_GRID):
        for v_snr in cfg.VAL_SNR_LIST:
            snr_t = torch.tensor(float(v_snr), device=device)
            W = model(H_dl_val, H_ul_val, snr_t, d_f, d_a, ci)
            _, r = criterion(W, H_dl_val, 10 ** (-v_snr / 10))
            table[(ci, v_snr)] = r
    return table


def print_val_table(table, step, pbar):
    lines = [f"📊 Step {step} | 验证矩阵 (bps/Hz):"]
    header = "    config        " + " | ".join(
        [f"{s:>4}dB" for s in cfg.VAL_SNR_LIST]
    ) + " |  avg"
    lines.append(header)
    for ci, (d_f, d_a) in enumerate(cfg.GAMMA_GRID):
        vals = [table[(ci, s)] for s in cfg.VAL_SNR_LIST]
        row = (f"    ({d_f:>2},{d_a:>3})      "
               + " | ".join([f"{v:6.2f}" for v in vals])
               + f" | {np.mean(vals):5.2f}")
        lines.append(row)
    overall = np.mean(list(table.values()))
    lines.append(f"    ───────── overall avg: {overall:.3f}")
    pbar.write("\n".join(lines))
    return overall


# ============================================================================
# 主训练逻辑
# ============================================================================
def main():
    torch.set_float32_matmul_precision('high')
    set_seed(cfg.SEED)
    device = setup_gpu()

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(f"🚀 [Experiment] {cfg.EXP_NAME}")
    print(f"📁 [Checkpoint] {cfg.SAVE_DIR}")
    print(f"🎛️ [γ Grid] {cfg.GAMMA_GRID}")
    print(f"🧪 [Unified Recipe] DIR_MODE={cfg.DIR_MODE} | "
          f"Ph1(dir={cfg.PH1_W_DIR}, mse={cfg.PH1_W_MSE}) → "
          f"Ph2(dir={cfg.PH2_W_DIR}, mse={cfg.PH2_W_MSE})")

    model = AdaptiveHybridPrecoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 [Model] AdaptiveHybrid | Params: {n_params / 1e6:.2f}M | "
          f"L={cfg.NUM_UNFOLD_LAYERS} | n_cfgs={len(cfg.GAMMA_GRID)}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR, total_steps=cfg.TOTAL_STEPS,
        pct_start=cfg.WARMUP_PCT, anneal_strategy='cos'
    )

    # ----- 断点续训 -----
    start_step = 1
    best_rate = -1.0
    if os.path.exists(cfg.SAVE_PATH):
        ckpt = torch.load(cfg.SAVE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step'] + 1
        best_rate = ckpt.get('best_rate', -1.0)
        print(f"✅ Resumed from step {start_step}, best overall: {best_rate:.3f}")

    criterion = SumRateLoss()
    channel_gen = CDLChannelGenerator_Feedback(
        Nt=cfg.ANTENNAS, Nsc=cfg.SUBCARRIERS,
        speed=cfg.SPEED, num_users=cfg.K_USERS
    ).to(device)

    # ----- 固定验证集 -----
    print("📊 Preparing validation set...")
    with torch.no_grad():
        h_dl_v, h_ul_v = channel_gen.generate_batch_data(cfg.VAL_BATCH)
        H_dl_val, H_ul_val = h_dl_v.to(device), h_ul_v.to(device)

    # ----- 训练循环 -----
    cache_data = []
    CACHE_SIZE = 30
    pbar = tqdm(range(start_step, cfg.TOTAL_STEPS + 1),
                desc="Adaptive-Hybrid", dynamic_ncols=True)

    for step in pbar:
        if not cache_data:
            with torch.no_grad():
                for _ in range(CACHE_SIZE):
                    d, u = channel_gen.generate_batch_data(cfg.BATCH_SIZE)
                    cache_data.append((d, u))

        H_dl_b, H_ul_b = cache_data.pop(0)
        H_dl_b, H_ul_b = H_dl_b.to(device), H_ul_b.to(device)

        # ===== 每步随机采样一个资源配置 (整 batch 共享) =====
        cfg_idx = np.random.randint(len(cfg.GAMMA_GRID))
        d_f, d_a = cfg.GAMMA_GRID[cfg_idx]

        model.train()
        optimizer.zero_grad(set_to_none=True)

        snr = get_snr(step)
        noise_pwr = 10 ** (-snr / 10)
        snr_t = torch.tensor(snr, device=device, dtype=torch.float32)

        W, H_hat = model(H_dl_b, H_ul_b, snr_t, d_f, d_a, cfg_idx)
        sr_loss, sr_val = criterion(W, H_dl_b, noise_pwr)
        dir_loss, mse_loss, cos_sim = alignment_losses(H_hat, H_dl_b)

        # ===== [P0] 统一两阶段配方, 所有 γ 共用 =====
        if step <= cfg.PHASE1_STEPS:
            w_dir, w_mse = cfg.PH1_W_DIR, cfg.PH1_W_MSE
            phase = "Ph1_Anchor"
        else:
            w_dir, w_mse = cfg.PH2_W_DIR, cfg.PH2_W_MSE
            phase = "Ph2_MaxSR"

        loss = sr_loss + w_dir * dir_loss + w_mse * mse_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            pbar.set_postfix({
                'Phase': phase,
                'cfg': f"({d_f},{d_a})",
                'SR': f"{sr_val:.2f}",
                'Cos': f"{cos_sim.item():.3f}",
                'MSE': f"{mse_loss.item():.4f}",
                'SNR': f"{snr:.1f}",
            })

        # ===== 验证: 全 (config × SNR) 矩阵 =====
        if step % cfg.VAL_INTERVAL == 0:
            table = validate(model, H_dl_val, H_ul_val, criterion, device)
            overall = print_val_table(table, step, pbar)

            if overall > best_rate:
                best_rate = overall
                pbar.write(f"✨ New best overall: {best_rate:.3f} → saved")
                torch.save({
                    'step': step,
                    'state': model.state_dict(),
                    'best_rate': best_rate,
                    'gamma_grid': cfg.GAMMA_GRID,
                    'val_table': {f"{k}": v for k, v in table.items()},
                }, cfg.BEST_PATH)

            torch.save({
                'step': step,
                'state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_rate': best_rate,
                'gamma_grid': cfg.GAMMA_GRID,
            }, cfg.SAVE_PATH)
            torch.cuda.empty_cache()

    print(f"✅ Training finished. Best overall avg: {best_rate:.3f}")


if __name__ == "__main__":
    main()