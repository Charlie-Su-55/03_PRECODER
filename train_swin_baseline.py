#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_swin_baseline.py
=========================
SwinCFNet + E2E ZF Baseline 训练
对标 4UE × 128SC 的 Pure FDMA 模式

注意: 复用主模型的 cfg, 但额外用 cfg.SWIN_* 那一组超参
      SWIN_CODEWORD_DIM 必须等于 DIM_FDMA_PER_USER * 2 才公平
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

from configs.exp_4ue_128sc import HybridPrecoderConfig as cfg
from models.swin_precoder import Swin_E2E_Precoder
from utils import CDLChannelGenerator_Feedback, setup_gpu


# ============================================================
# Sum Rate Loss (跟主模型完全一致, 保证公平)
# ============================================================
class SumRateLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, W, H_dl_true, noise_power):
        h_eff = torch.einsum('bkns,bnjs->bkjs', H_dl_true.conj(), W)
        power = h_eff.abs() ** 2
        sig_pwr = torch.stack([power[:, i, i, :] for i in range(power.shape[1])], dim=1)
        total_pwr = power.sum(dim=2)
        int_pwr = total_pwr - sig_pwr
        sinr = sig_pwr / (int_pwr + noise_power + self.eps)
        rate = torch.log2(1.0 + sinr)
        avg_rate = rate.sum(dim=1).mean()
        return -avg_rate, avg_rate.item()


# ============================================================
# SNR 课程学习 (跟主模型一致)
# ============================================================
def get_snr(step, cfg):
    if step <= cfg.STAGE1_STEPS:
        return np.random.uniform(cfg.STAGE1_SNR[0], cfg.STAGE1_SNR[1])
    elif step <= cfg.STAGE2_STEPS:
        return np.random.uniform(cfg.STAGE2_SNR[0], cfg.STAGE2_SNR[1])
    else:
        return np.random.uniform(cfg.STAGE3_SNR[0], cfg.STAGE3_SNR[1])


# ============================================================
# 主训练
# ============================================================
def main():
    torch.set_float32_matmul_precision('high')
    device = setup_gpu()

    # 检查 config 是否对齐 Pure FDMA 模式
    assert cfg.DIM_AIRCOMP_SHARED == 0, \
        "Swin baseline 只对标 Pure FDMA 模式, 请把 DIM_AIRCOMP_SHARED 设为 0"
    assert cfg.DIM_FDMA_PER_USER > 0, \
        "Swin baseline 需要 FDMA 反馈, 请设置 DIM_FDMA_PER_USER > 0"

    # 路径
    save_dir = f"ckp/swin_baseline_{cfg.K_USERS}ue_{cfg.SUBCARRIERS}sc_fdma{cfg.DIM_FDMA_PER_USER}"
    os.makedirs(save_dir, exist_ok=True)
    BEST_PATH = os.path.join(save_dir, "swin_best.pth")
    LAST_PATH = os.path.join(save_dir, "swin_latest.pth")

    print(f"🎯 [Swin Baseline] {cfg.K_USERS}UE × {cfg.SUBCARRIERS}SC, "
          f"FDMA D_f={cfg.DIM_FDMA_PER_USER}")
    print(f"📁 Save dir: {save_dir}")
    print(f"🔢 SWIN_CODEWORD_DIM={cfg.SWIN_CODEWORD_DIM} "
          f"(对齐 D_f*2={cfg.DIM_FDMA_PER_USER * 2})")

    # 模型
    model = Swin_E2E_Precoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 [Swin] Params: {n_params/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR, total_steps=cfg.TOTAL_STEPS,
        pct_start=cfg.WARMUP_PCT, anneal_strategy='cos'
    )

    # 断点续训
    start_step = 1
    best_rate = -1.0
    if os.path.exists(LAST_PATH):
        ckpt = torch.load(LAST_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step'] + 1
        best_rate = ckpt.get('best_rate', -1.0)
        print(f"✅ Resumed from step {start_step}, Best: {best_rate:.3f}")

    criterion = SumRateLoss()
    channel_gen = CDLChannelGenerator_Feedback(
        Nt=cfg.ANTENNAS, Nsc=cfg.SUBCARRIERS,
        speed=cfg.SPEED, num_users=cfg.K_USERS
    )

    # 验证集
    print("📊 Preparing validation set...")
    with torch.no_grad():
        h_dl_v, h_ul_v = channel_gen.generate_batch_data(batch_size=64)
        H_dl_val, H_ul_val = h_dl_v.to(device), h_ul_v.to(device)

    # 训练循环
    cache_data = []
    CACHE_SIZE = 30
    pbar = tqdm(range(start_step, cfg.TOTAL_STEPS + 1),
                desc="Swin-Baseline", dynamic_ncols=True)

    for step in pbar:
        if not cache_data:
            with torch.no_grad():
                for _ in range(CACHE_SIZE):
                    d, u = channel_gen.generate_batch_data(cfg.BATCH_SIZE)
                    cache_data.append((d, u))

        H_dl_b, H_ul_b = cache_data.pop(0)
        H_dl_b, H_ul_b = H_dl_b.to(device), H_ul_b.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        snr = get_snr(step, cfg)
        noise_pwr = 10 ** (-snr / 10)
        snr_tensor = torch.tensor(snr, device=device, dtype=torch.float32)

        W = model(H_dl_b, H_ul_b, snr_db=snr_tensor)
        loss, sr_val = criterion(W, H_dl_b, noise_pwr)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0:
            pbar.set_postfix({
                'SR': f"{sr_val:.2f}",
                'SNR': f"{snr:.1f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.1e}"
            })

        if step % cfg.VAL_INTERVAL == 0:
            model.eval()
            val_results = {}
            with torch.no_grad():
                for v_snr in cfg.VAL_SNR_LIST:
                    snr_v = torch.tensor(v_snr, device=device, dtype=torch.float32)
                    W_v = model(H_dl_val, H_ul_val, snr_db=snr_v)
                    _, rate_v = criterion(W_v, H_dl_val, 10 ** (-v_snr / 10))
                    val_results[v_snr] = rate_v

            avg_v = float(np.mean(list(val_results.values())))
            snr_str = " | ".join([f"{k}dB:{v:.2f}" for k, v in val_results.items()])
            info_str = f"🚀 Step {step} | Avg:{avg_v:.3f} | {snr_str}"

            if avg_v > best_rate:
                best_rate = avg_v
                pbar.write(f"✨ {info_str} (New Best!)")
                torch.save(
                    {'step': step, 'state': model.state_dict(),
                     'best_rate': best_rate}, BEST_PATH
                )
            else:
                pbar.write(f"   {info_str}")

            torch.save({
                'step': step, 'state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_rate': best_rate
            }, LAST_PATH)
            torch.cuda.empty_cache()

    print(f"✅ Training finished. Best avg rate: {best_rate:.3f}")


if __name__ == "__main__":
    main()