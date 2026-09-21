#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snr_adaptive_4ue.py — 新版 Figure 4: 单 checkpoint 自适应模型 vs 全部外部 baseline
===============================================================================
对比对象:
  - Adaptive (单 ckpt, 5 个 (D_f, D_a) 档位)   ← 本文方案
  - Perfect CSI ZF (上界)
  - CsiNet+ / Swin-CFnet (深度 FDMA baseline)
  - 5G NR Type II (L=2 / L=4)

评估协议与 snr_sweep_4ue.py 完全一致 (TEST_SEED / NUM_BATCHES / per-sample 口径),
数字可与已有结果直接对比。

额外输出: 配对显著性检验 (同一批信道上 hybrid 档位 vs 纯 AirComp 的逐样本差值),
用于回答 "为什么需要 hybrid 而不是永远纯 AirComp"。

用法: python snr_adaptive_4ue.py
输出: results/adaptive/snr_adaptive_4ue.{png,pdf,json,csv} + terminal 表格
"""
import os
import json
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNR_LIST = [-5, 0, 5, 10, 15, 20, 25, 30]
NUM_BATCHES = 50
TEST_SEED = 20261111
OUT_DIR = os.path.join("results", "adaptive")
os.makedirs(OUT_DIR, exist_ok=True)

ADAPTIVE_CKPT_CANDIDATES = [
    "ckp/adaptive_uwmmse_4ue_128sc_dtot128_sb4_unified_50k_seed42/adaptive_best.pth",
    "ckp/adaptive_uwmmse_4ue_128sc_dtot128_sb4_unified_seed42/adaptive_best.pth",
]


# ============================================================
# 1. Per-sample sum-rate (与 snr_sweep_4ue.py 完全一致)
# ============================================================
@torch.no_grad()
def compute_sum_rate_per_sample(W, H_dl, snr_db):
    sigma2 = 10.0 ** (-snr_db / 10.0)
    if W.shape[1] == H_dl.shape[2] and W.shape[2] == H_dl.shape[1]:
        W = W.permute(0, 2, 1, 3)
    HW = torch.einsum('bkns,bjns->bkjs', H_dl.conj(), W)
    pw = HW.abs() ** 2
    K = pw.shape[1]
    eye = torch.eye(K, device=W.device, dtype=pw.dtype)
    signal = (pw * eye.view(1, K, K, 1)).sum(dim=2)
    interference = (pw * (1 - eye).view(1, K, K, 1)).sum(dim=2)
    sinr = signal / (interference + sigma2)
    per_user_rate = torch.log2(1.0 + sinr)
    return per_user_rate.sum(dim=1).mean(dim=1).detach().cpu().numpy()


def summarize(samples):
    n = len(samples)
    std = float(np.std(samples, ddof=1))
    sem = std / np.sqrt(n)
    return {"mean": float(np.mean(samples)), "std": std,
            "sem": sem, "ci95": 1.96 * sem, "n": n}


# ============================================================
# 2. 模型加载
# ============================================================
def _load_state(model, ckpt_path, strict=False):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    for key in ('state', 'model', 'state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    model.load_state_dict(state, strict=strict)
    return model


class _AdaptiveAtConfig(torch.nn.Module):
    """ 把共享的 adaptive 核心包成固定 (d_f, d_a) 档位, 适配统一评估接口 """
    def __init__(self, core, d_f, d_a, cfg_idx):
        super().__init__()
        self.core = core
        self.d_f, self.d_a, self.ci = d_f, d_a, cfg_idx

    def forward(self, H_dl, H_ul, snr_db):
        snr_t = torch.tensor(float(snr_db), device=H_dl.device)
        return self.core(H_dl, H_ul, snr_t, self.d_f, self.d_a, self.ci)


def build_adaptive_core():
    from train_adaptive import AdaptiveConfig as ACfg
    from models.adaptive_hybrid import AdaptiveHybridPrecoder
    ckpt = next((p for p in ADAPTIVE_CKPT_CANDIDATES if os.path.exists(p)), None)
    if ckpt is None:
        raise FileNotFoundError(
            f"未找到 adaptive ckpt, 尝试过: {ADAPTIVE_CKPT_CANDIDATES}")
    core = AdaptiveHybridPrecoder(ACfg).to(DEVICE)
    _load_state(core, ckpt, strict=True)
    core.eval()
    print(f"✅ adaptive ckpt: {ckpt}")
    return core, ACfg


def build_csinet():
    from configs.exp_4ue_128sc import HybridPrecoderConfig
    from models.csinet_plus_precoder import CsiNetPlus_E2E_Precoder
    cfg = HybridPrecoderConfig()
    cfg.DIM_FDMA_PER_USER = 32
    model = CsiNetPlus_E2E_Precoder(cfg).to(DEVICE)
    _load_state(model, "ckp/csinet_baseline_4ue_128sc_fdma32/csinet_best.pth")
    model.eval()
    return model


def build_swin():
    from configs.exp_4ue_128sc import HybridPrecoderConfig
    from models.swin_precoder import Swin_E2E_Precoder
    cfg = HybridPrecoderConfig()
    cfg.DIM_FDMA_PER_USER = 32
    cfg.SWIN_CODEWORD_DIM = 64
    model = Swin_E2E_Precoder(cfg).to(DEVICE)
    _load_state(model, "ckp/swin_baseline_4ue_128sc_fdma32/swin_best.pth")
    model.eval()
    return model


def build_type2(L):
    from configs.exp_4ue_128sc import HybridPrecoderConfig
    from models.type2_codebook import Type2CodebookBaseline
    cfg = HybridPrecoderConfig()
    model = Type2CodebookBaseline(cfg, L=L).to(DEVICE)
    model.eval()
    return model


class PerfectCSI_ZF_Precoder(torch.nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, H_dl, H_ul=None, snr_db=None):
        H_eff = H_dl.conj()
        H_eff_p = H_eff.permute(0, 3, 1, 2)
        W_zf = torch.linalg.pinv(H_eff_p)
        W_zf = W_zf.permute(0, 3, 2, 1)
        power = (W_zf.abs() ** 2).sum(dim=(1, 2), keepdim=True)
        return W_zf / torch.sqrt(power + 1e-12)


# ============================================================
# 3. 测试集
# ============================================================
def build_test_batches(cfg):
    from utils import CDLChannelGenerator_Feedback
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    gen = CDLChannelGenerator_Feedback(
        Nt=cfg.ANTENNAS, Nsc=cfg.SUBCARRIERS,
        carrier_freq=getattr(cfg, 'CARRIER_FREQ', 3.5e9),
        speed=getattr(cfg, 'SPEED', 1.0),
        num_users=cfg.K_USERS,
    ).to(DEVICE)
    batches = []
    for _ in range(NUM_BATCHES):
        H_dl, H_ul = gen.generate_batch_data(
            batch_size=cfg.BATCH_SIZE, device=DEVICE)
        batches.append((H_dl, H_ul))
    return batches


# ============================================================
# 4. 评估 (返回 per-sample 数组, 供配对检验)
# ============================================================
@torch.no_grad()
def eval_model(model, batches, snr_db, needs_hul=True):
    chunks = []
    for H_dl, H_ul in batches:
        if needs_hul:
            out = model(H_dl, H_ul, snr_db=snr_db)
        else:
            out = model(H_dl, snr_db=snr_db)
        W = out[0] if isinstance(out, tuple) else out
        chunks.append(compute_sum_rate_per_sample(W, H_dl, snr_db))
    return np.concatenate(chunks)


# ============================================================
# 5. 终端表格 / 配对检验
# ============================================================
def print_results_table(all_results, order):
    print(f"\n{'=' * 100}")
    print(f"汇总表 (mean bps/Hz, seed={TEST_SEED}, "
          f"n={NUM_BATCHES} batches)")
    print(f"{'=' * 100}")
    header = f"  {'model':<22}" + " | ".join([f"{s:>5}dB" for s in SNR_LIST])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in order:
        if name not in all_results:
            continue
        cells = [f"{all_results[name][str(s)]['mean']:7.2f}" for s in SNR_LIST]
        print(f"  {name:<22}" + " | ".join(cells))


def print_paired_tests(samples_store, pure_key, hybrid_keys):
    """ 配对检验: 同一批信道上 hybrid − pure AirComp 的逐样本差值 """
    if pure_key not in samples_store:
        return
    print(f"\n{'=' * 100}")
    print(f"配对检验: hybrid − 纯AirComp {pure_key} "
          f"(同一信道逐样本差, 正数=hybrid更好, ±为95% CI)")
    print(f"{'=' * 100}")
    for hk in hybrid_keys:
        if hk not in samples_store:
            continue
        cells = []
        for s in SNR_LIST:
            d = samples_store[hk][s] - samples_store[pure_key][s]
            m, ci = float(np.mean(d)), 1.96 * np.std(d, ddof=1) / np.sqrt(len(d))
            star = "*" if abs(m) > ci else " "
            cells.append(f"{m:+6.2f}±{ci:4.2f}{star}")
        print(f"  {hk:<22}" + " | ".join(cells))
    print("  (* = 差值的 95% CI 不跨零, 即统计显著)")


# ============================================================
# 6. 绘图
# ============================================================
STYLE = {
    "Perfect CSI ZF":     dict(marker="*", color="#7f7f7f", ls="-.", lw=2.0, z=1),
    "Adaptive (0,128)":   dict(marker="o", color="#d62728", ls="-",  lw=2.5, z=10),
    "Adaptive (8,96)":    dict(marker="d", color="#e377c2", ls="-",  lw=2.0, z=9),
    "Adaptive (16,64)":   dict(marker="X", color="#ff7f0e", ls="-",  lw=2.0, z=8),
    "Adaptive (24,32)":   dict(marker="p", color="#8c564b", ls="-",  lw=2.0, z=8),
    "Adaptive (32,0)":    dict(marker="P", color="#9467bd", ls="-",  lw=2.0, z=7),
    "CsiNet+":            dict(marker="s", color="#1f77b4", ls="--", lw=1.5, z=5),
    "Swin-CFnet":         dict(marker="^", color="#17becf", ls="--", lw=1.5, z=5),
    "Type II L=2":        dict(marker="v", color="#2ca02c", ls=":",  lw=1.5, z=3),
    "Type II L=4":        dict(marker="D", color="#bcbd22", ls=":",  lw=1.5, z=3),
}


def plot_figure(all_results):
    plt.figure(figsize=(10, 7))
    for name, st in STYLE.items():
        if name not in all_results:
            continue
        snrs = sorted(int(s) for s in all_results[name].keys())
        means = [all_results[name][str(s)]["mean"] for s in snrs]
        cis = [all_results[name][str(s)]["ci95"] for s in snrs]
        plt.errorbar(snrs, means, yerr=cis, label=name, markersize=8,
                     marker=st["marker"], color=st["color"], linestyle=st["ls"],
                     linewidth=st["lw"], zorder=st["z"],
                     capsize=3, capthick=1.2, elinewidth=1.0)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Feedback SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('Sum-Rate (bps/Hz)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.02, 1),
               borderaxespad=0.)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(OUT_DIR, f"snr_adaptive_4ue.{ext}")
        plt.savefig(p, dpi=300, bbox_inches='tight')
    print(f"\n🖼️  saved → {OUT_DIR}/snr_adaptive_4ue.{{png,pdf}}")


# ============================================================
# 7. 主流程
# ============================================================
def main():
    # ----- adaptive 核心 (只加载一次, 5 个档位共享) -----
    core, acfg = build_adaptive_core()
    gamma_grid = list(acfg.GAMMA_GRID)

    print(f"[cache] building test batches (seed={TEST_SEED}, "
          f"{NUM_BATCHES} × {acfg.BATCH_SIZE} = "
          f"{NUM_BATCHES * acfg.BATCH_SIZE} samples)...")
    batches = build_test_batches(acfg)

    # ----- 注册全部待评模型 -----
    registry = {}
    for ci, (df, da) in enumerate(gamma_grid):
        registry[f"Adaptive ({df},{da})"] = (
            _AdaptiveAtConfig(core, df, da, ci), True)
    registry["Perfect CSI ZF"] = (PerfectCSI_ZF_Precoder(acfg.K_USERS), True)
    for name, builder in [("CsiNet+", build_csinet),
                          ("Swin-CFnet", build_swin),
                          ("Type II L=2", lambda: build_type2(2)),
                          ("Type II L=4", lambda: build_type2(4))]:
        try:
            registry[name] = (builder(), True)
        except Exception as e:
            print(f"  ! skip {name}: {e}")

    # ----- 评估 -----
    all_results = {}
    samples_store = {}     # 仅 adaptive 档位保留逐样本, 供配对检验
    for name, (model, needs_hul) in registry.items():
        per_snr = {}
        per_sample = {}
        for snr in tqdm(SNR_LIST, desc=f"  {name:<22}", leave=False):
            try:
                samples = eval_model(model, batches, snr, needs_hul)
            except TypeError:
                # 个别 baseline forward 签名不带 snr_db / H_ul 的兜底
                samples = eval_model(model, batches, snr, needs_hul=False)
            per_snr[str(snr)] = summarize(samples)
            if name.startswith("Adaptive"):
                per_sample[snr] = samples
        all_results[name] = per_snr
        if per_sample:
            samples_store[name] = per_sample
        print(f"  ✓ {name:<22} @25dB: "
              f"{per_snr['25']['mean']:6.2f} ± {per_snr['25']['ci95']:.2f}")
        if not name.startswith("Adaptive"):
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ----- 终端输出 -----
    order = list(STYLE.keys())
    print_results_table(all_results, order)
    pure_key = f"Adaptive (0,128)"
    hybrid_keys = [f"Adaptive ({df},{da})"
                   for (df, da) in gamma_grid if da not in (0, 128)] \
                  + ["Adaptive (32,0)"]
    print_paired_tests(samples_store, pure_key, hybrid_keys)

    # ----- 存档 -----
    with open(os.path.join(OUT_DIR, "snr_adaptive_4ue.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "snr_adaptive_4ue.csv"), "w",
              newline='') as f:
        w = csv.writer(f)
        w.writerow(["model", "snr_db", "mean", "std", "sem", "ci95", "n"])
        for name, d in all_results.items():
            for snr_str, v in d.items():
                w.writerow([name, snr_str, v["mean"], v["std"],
                            v["sem"], v["ci95"], v["n"]])
    print(f"💾 saved → {OUT_DIR}/snr_adaptive_4ue.{{json,csv}}")

    plot_figure(all_results)
    print("\n[done]")


if __name__ == "__main__":
    main()