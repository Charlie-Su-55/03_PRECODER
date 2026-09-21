"""
diagnostic_eval.py  —  Phase Transition Evaluation WITH ERROR BARS
======================================================================
K ∈ {2, 4, 8}  ×  (D_f, D_a) sweep, independent test set (fixed seed).
Per-sample sum-rate statistics → mean / std / SEM / 95% CI.
Includes paired significance test for the K=4 N-shape (peak vs valley).

Outputs (under results/phase_transition/):
  phase_transition_data.json      : statistics (mean/std/sem/ci95)
  phase_transition_data.csv       : flat version for paper tables
  phase_transition_samples.npz    : raw per-sample arrays (reproducibility)
  fig_phase_transition_25dB.{pdf,png}      : MAIN plot (25 dB, 95% CI bars)
  fig_phase_transition_multiSNR.{pdf,png}  : 2×2 panel, all SNRs
  SIGNIFICANCE.md                 : paired test — is the N-shape real?
======================================================================
"""
import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.unfolded_wmmse_precoder import MIMO_UnfoldedWMMSE_Precoder
from utils import CDLChannelGenerator_Feedback


# ======================================================================
# 1. Global config
# ======================================================================
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SEED       = 20261111            # independent of training seed (20260513)
NUM_BATCHES     = 20                  # 20 × 32 = 640 channel realizations
TEST_BATCH_SIZE = 32
EVAL_SNR_LIST   = [0.0, 10.0, 20.0, 25.0]   # all SNRs evaluated in one pass
MAIN_SNR        = 25.0                # SNR used for the main phase-transition figure
RESULTS_DIR     = "results/phase_transition"
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG_MATRIX = {
    2: [(64, 0), (48, 32), (32, 64), (16, 96), (0, 128)],
    4: [(32, 0), (24, 32), (16, 64), (8, 96), (0, 128)],
    8: [(16, 0), (12, 32), (8, 64), (4, 96), (0, 128)],
}


class EvalConfig:
    ANTENNAS = 32
    SUBCARRIERS = 128
    NUM_SUBBANDS = 4
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 4
    NUM_UNFOLD_LAYERS = 5
    GNN_AGG_DIM = 48
    TOTAL_POWER = 1.0
    BATCH_SIZE = TEST_BATCH_SIZE
    CARRIER_FREQ = 3.5e9
    SPEED = 1.0
    K_USERS = None
    DIM_FDMA_PER_USER = None
    DIM_AIRCOMP_SHARED = None
    DROPOUT = 0.1


# ======================================================================
# 2. Per-SAMPLE sum-rate  (【ERROR BAR】 key change: no .mean() over batch)
# ======================================================================
@torch.no_grad()
def compute_sum_rate_per_sample(W, H_dl, snr_db):
    """Returns a numpy array of shape [B] — one sum-rate per channel realization."""
    sigma2 = 10.0 ** (-snr_db / 10.0)
    if W.shape[1] == H_dl.shape[2] and W.shape[2] == H_dl.shape[1]:
        W = W.permute(0, 2, 1, 3)

    HW = torch.einsum('bkns,bjns->bkjs', H_dl.conj(), W)
    pw = HW.abs() ** 2

    K = pw.shape[1]
    eye = torch.eye(K, device=W.device, dtype=pw.dtype)
    signal       = (pw * eye.view(1, K, K, 1)).sum(dim=2)
    interference = (pw * (1 - eye).view(1, K, K, 1)).sum(dim=2)

    sinr = signal / (interference + sigma2)
    per_user_rate = torch.log2(1.0 + sinr)              # [B, K, Nsc]
    per_sample = per_user_rate.sum(dim=1).mean(dim=1)   # [B]  sum over K, avg over Nsc
    return per_sample.detach().cpu().numpy()

@torch.no_grad()
def compute_fair_perfect_csi_zf(H_dl):
    """
    Fair Perfect CSI ZF (Total Power = 1.0 per subcarrier)
    H_dl: [B, K, Nt, Nsc]
    Returns W: [B, K, Nt, Nsc]
    """
    # 1. 提取等效信道 C = H*，并正确变换维度为 [B, Nsc, K, Nt] 
    H_eff_p = H_dl.conj().permute(0, 3, 1, 2)
    
    # 2. 批量求伪逆，得到维度 [B, Nsc, Nt, K]
    W_raw = torch.linalg.pinv(H_eff_p)
    
    # 3. 转回你 Einsum 计算所需的格式 [B, K, Nt, Nsc]
    W_zf = W_raw.permute(0, 3, 2, 1)
    
    # 4. 严格的公平功率归一化 (总功率 = 1.0)
    # 对每个子载波的所有天线 (dim=2) 和所有用户 (dim=1) 求总功率
    power = (W_zf.abs() ** 2).sum(dim=(1, 2), keepdim=True)
    
    # 5. 除以 sqrt(power) 保证严格的 1.0 功率
    return W_zf / torch.sqrt(power + 1e-12)

def summarize(samples):
    """samples: 1-D np array of per-sample sum-rates → dict of statistics."""
    n = len(samples)
    mean = float(np.mean(samples))
    std  = float(np.std(samples, ddof=1))               # unbiased sample std
    sem  = std / np.sqrt(n)                             # standard error of the mean
    ci95 = 1.96 * sem                                   # 95% confidence half-width
    return {"mean": mean, "std": std, "sem": sem, "ci95": ci95, "n": n}


# ======================================================================
# 3. Unified test set (seed-locked per K)
# ======================================================================
def build_test_dataset(K):
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    channel_gen = CDLChannelGenerator_Feedback(
        Nt=EvalConfig.ANTENNAS,
        Nsc=EvalConfig.SUBCARRIERS,
        carrier_freq=EvalConfig.CARRIER_FREQ,
        speed=EvalConfig.SPEED,
        num_users=K,
    ).to(DEVICE)
    print(f"  [Dataset] Generating {NUM_BATCHES * TEST_BATCH_SIZE} samples for K={K}...")
    batches = []
    for _ in range(NUM_BATCHES):
        H_dl, H_ul = channel_gen.generate_batch_data(
            batch_size=TEST_BATCH_SIZE, device=DEVICE)
        batches.append((H_dl, H_ul))
    return batches


# ======================================================================
# 4. Main evaluation loop
# ======================================================================
def main():
    print("Starting Phase-Transition Evaluation (with error bars)...")

    # stats[K][snr_str][cfg_idx]   = {mean, std, sem, ci95, n}
    # raw_samples[K][snr_str][cfg_idx] = np.array([640])  (kept for npz + sig test)
    stats = {K: {str(s): [None] * len(cfgs) for s in EVAL_SNR_LIST}
             for K, cfgs in CONFIG_MATRIX.items()}
    raw_samples = {K: {str(s): [None] * len(cfgs) for s in EVAL_SNR_LIST}
                   for K, cfgs in CONFIG_MATRIX.items()}

    for K, configs in CONFIG_MATRIX.items():
        print(f"\n==================== Evaluating K = {K} ====================")
        test_data = build_test_dataset(K)

        # --- 【计算当前 K 的绝对公平 Perfect CSI ZF 上界】 ---
        zf_samples_list = []
        with torch.no_grad():
            for H_dl, H_ul in test_data:
                W_zf = compute_fair_perfect_csi_zf(H_dl)
                zf_samples_list.append(compute_sum_rate_per_sample(W_zf, H_dl, MAIN_SNR))
        zf_samples = np.concatenate(zf_samples_list)
        zf_st = summarize(zf_samples)
        print(f" 🌟 Fair Perfect CSI ZF @ {MAIN_SNR:4.0f}dB : {zf_st['mean']:6.3f}  ± {zf_st['ci95']:.3f} (95% CI)")
        # --- 【计算结束】 ---

        for cfg_idx, (df, da) in enumerate(configs):
            cfg = EvalConfig()
            cfg.K_USERS = K
            cfg.DIM_FDMA_PER_USER = df
            cfg.DIM_AIRCOMP_SHARED = da

            exp_name = f"uwmmse_{K}ue_128sc_fdma{df}_ac{da}_L5_sb4"
            ckpt_path = f"ckp/{exp_name}/uwmmse_best.pth"

            if not os.path.exists(ckpt_path):
                print(f"  MISSING: {exp_name} -> filling NaN")
                for s in EVAL_SNR_LIST:
                    stats[K][str(s)][cfg_idx] = {
                        "mean": float("nan"), "std": float("nan"),
                        "sem": float("nan"), "ci95": float("nan"), "n": 0}
                continue

            model = MIMO_UnfoldedWMMSE_Precoder(cfg).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            state = ckpt['state'] if 'state' in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            model.eval()

            # --- evaluate this model at every SNR (loaded once, cheap) ---
            for snr in EVAL_SNR_LIST:
                snr_tensor = torch.tensor(snr, device=DEVICE, dtype=torch.float32)
                sample_chunks = []
                with torch.no_grad():
                    for H_dl, H_ul in test_data:
                        try:
                            out = model(H_dl, H_ul, snr_db=snr_tensor)
                        except TypeError:
                            out = model(H_dl, snr_db=snr_tensor)
                        W = out[0] if isinstance(out, tuple) else out
                        
                        # if snr == MAIN_SNR and cfg_idx == 0:
                        #     tp = (W.abs() ** 2).sum(dim=(1, 2)).mean().item()
                        #     print(f"    [power check] model W total power = {tp:.4f}")
                        
                        sample_chunks.append(
                            compute_sum_rate_per_sample(W, H_dl, snr))
                samples = np.concatenate(sample_chunks)        # [640]
                st = summarize(samples)
                stats[K][str(snr)][cfg_idx] = st
                raw_samples[K][str(snr)][cfg_idx] = samples

                tag = " <-- MAIN" if snr == MAIN_SNR else ""
                print(f"  (Df={df:2d}, Da={da:3d}) @ {snr:4.0f}dB :  "
                      f"{st['mean']:6.3f}  ± {st['ci95']:.3f} (95% CI){tag}")

            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ---- Save raw data ----
    _save_json_csv(stats)
    _save_npz(raw_samples)

    # ---- Significance test on K=4 N-shape ----
    significance_report(raw_samples, stats)

    # ---- Plots ----
    plot_main(stats, MAIN_SNR)
    plot_multi_snr(stats)

    print(f"\n[done] all outputs under: {RESULTS_DIR}/")


# ======================================================================
# 5. Save helpers
# ======================================================================
def _save_json_csv(stats):
    with open(os.path.join(RESULTS_DIR, "phase_transition_data.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "phase_transition_data.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "D_f", "D_a", "snr_db",
                    "mean", "std", "sem", "ci95", "n"])
        for K, configs in CONFIG_MATRIX.items():
            for snr in EVAL_SNR_LIST:
                for cfg_idx, (df, da) in enumerate(configs):
                    st = stats[K][str(snr)][cfg_idx]
                    w.writerow([K, df, da, snr, st["mean"], st["std"],
                                st["sem"], st["ci95"], st["n"]])
    print(f"[save] -> phase_transition_data.{{json,csv}}")


def _save_npz(raw_samples):
    flat = {}
    for K, configs in CONFIG_MATRIX.items():
        for snr in EVAL_SNR_LIST:
            for cfg_idx, (df, da) in enumerate(configs):
                arr = raw_samples[K][str(snr)][cfg_idx]
                if arr is not None:
                    flat[f"K{K}_df{df}_da{da}_snr{int(snr)}"] = arr
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "phase_transition_samples.npz"), **flat)
    print(f"[save] -> phase_transition_samples.npz ({len(flat)} arrays)")


# ======================================================================
# 6. Significance test — is the K=4 valley statistically real?
# ======================================================================
def significance_report(raw_samples, stats):
    lines = ["# K=4 N-Shape — Paired Significance Test",
             f"_Test set: {NUM_BATCHES*TEST_BATCH_SIZE} channels, "
             f"seed={TEST_SEED}, SNR={int(MAIN_SNR)} dB._",
             "",
             "Configs are evaluated on the **same** channel realizations, so "
             "differences are tested pairwise (higher statistical power).",
             ""]

    cfgs = CONFIG_MATRIX[4]
    snr_key = str(MAIN_SNR)
    s = raw_samples[4][snr_key]
    if any(x is None for x in s):
        lines.append("⚠️  Some K=4 checkpoints missing — test skipped.")
        with open(os.path.join(RESULTS_DIR, "SIGNIFICANCE.md"), "w") as f:
            f.write("\n".join(lines))
        return

    def paired(a_idx, b_idx):
        a, b = s[a_idx], s[b_idx]
        diff = a - b
        md = float(np.mean(diff))
        sem = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
        ci = 1.96 * sem
        t = md / sem if sem > 0 else float("inf")
        return md, ci, t

    # idx: 0=(32,0) 1=(24,32) 2=(16,64) 3=(8,96) 4=(0,128)
    tests = [
        ("Local peak (24,32)  >  Valley (16,64)", 1, 2),
        ("Recovery  (8,96)    >  Valley (16,64)", 3, 2),
        ("Local peak (24,32)  >  FDMA floor (32,0)", 1, 0),
    ]
    lines.append("| Comparison | Mean diff | 95% CI | t | Significant? |")
    lines.append("|---|---|---|---|---|")
    for desc, ai, bi in tests:
        md, ci, t = paired(ai, bi)
        sig = "✅ YES" if abs(md) > ci else "❌ no"
        lines.append(f"| {desc} | {md:+.3f} | ±{ci:.3f} | {t:+.1f} | {sig} |")

    lines.append("")
    lines.append("**Interpretation:** if every comparison is significant, the "
                 "non-monotonic N-shape is a real structural effect, not "
                 "sampling noise — safe to claim in the paper.")

    with open(os.path.join(RESULTS_DIR, "SIGNIFICANCE.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[save] -> SIGNIFICANCE.md")


# ======================================================================
# 7. Plotting
# ======================================================================
_PLT_RC = {
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 14, 'font.size': 12, 'legend.fontsize': 10,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
}

_K_STYLE = {
    2: dict(color='#1f77b4', marker='s', ms=8,  lw=2.5,
            label=r'$K=2$ (Abundant: Monotonic)'),
    4: dict(color='#d62728', marker='o', ms=9,  lw=3.2,
            label=r'$K=4$ (Scarce: N-Shape Phase Transition)'),
    8: dict(color='#2ca02c', marker='^', ms=9,  lw=2.5,
            label=r'$K=8$ (Starved: Monotonic, Anchor-Free)'),
}

_X_RATIO = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_X_TICKLABELS = ['0\n(Pure FDMA)', '0.25', '0.50', '0.75', '1.0\n(Pure AirComp)']


def _series(stats, K, snr):
    """Return (means, ci95s) arrays for a given K and SNR."""
    row = stats[K][str(snr)]
    means = np.array([r["mean"] for r in row])
    ci95s = np.array([r["ci95"] for r in row])
    return means, ci95s


def plot_main(stats, snr):
    plt.rcParams.update(_PLT_RC)
    fig, ax = plt.subplots(figsize=(8, 6))

    for K, st in _K_STYLE.items():
        means, ci95s = _series(stats, K, snr)
        if np.all(np.isnan(means)):
            continue
        ax.errorbar(_X_RATIO, means, yerr=ci95s,
                    color=st['color'], marker=st['marker'], markersize=st['ms'],
                    linewidth=st['lw'], mfc='white', markeredgewidth=2,
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    label=st['label'])

    # Annotate K=4 local optimum
    m4, _ = _series(stats, 4, snr)
    if not np.isnan(m4[1]):
        ax.annotate('Local Optimum\n(Anchor-Residual Sweet Spot)',
                    xy=(0.25, m4[1]), xytext=(0.07, m4[1] + 2.5),
                    arrowprops=dict(facecolor='black', shrink=0.05,
                                    width=1.5, headwidth=6),
                    fontsize=10, fontweight='bold', color='#d62728')
    if not np.isnan(m4[2]):
        ax.annotate('Valley',
                    xy=(0.50, m4[2]), xytext=(0.50, m4[2] - 3.0),
                    arrowprops=dict(facecolor='black', shrink=0.05,
                                    width=1.5, headwidth=6),
                    fontsize=10, fontweight='bold', color='#d62728', ha='center')

    ax.set_xlabel(r'AirComp Resource Ratio $\gamma = D_a / D_{\mathrm{tot}}$',
                  fontweight='bold')
    ax.set_ylabel(f'Sum-Rate @ {int(snr)} dB (bps/Hz)', fontweight='bold')
    ax.set_xticks(_X_RATIO)
    ax.set_xticklabels(_X_TICKLABELS)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')

    # caption note about error bars
    ax.text(0.99, 0.02, 'Error bars: 95% CI ($n=640$)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='#555', style='italic')

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        p = os.path.join(RESULTS_DIR, f"fig_phase_transition_{int(snr)}dB.{ext}")
        plt.savefig(p, format=ext, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plot] -> fig_phase_transition_{int(snr)}dB.{{pdf,png}}")


def plot_multi_snr(stats):
    plt.rcParams.update(_PLT_RC)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    def _interior_valleys(arr):
        """Indices of interior local minima (strictly below both neighbors)."""
        out = []
        for i in range(1, len(arr) - 1):
            trio = arr[i - 1:i + 2]
            if np.any(np.isnan(trio)):
                continue
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                out.append(i)
        return out

    for ax, snr in zip(axes, EVAL_SNR_LIST):
        for K, st in _K_STYLE.items():
            means, ci95s = _series(stats, K, snr)
            if np.all(np.isnan(means)):
                continue
            ax.errorbar(_X_RATIO, means, yerr=ci95s,
                        color=st['color'], marker=st['marker'],
                        markersize=st['ms'] - 1, linewidth=st['lw'] - 0.6,
                        mfc='white', markeredgewidth=1.8,
                        capsize=3, capthick=1.2, elinewidth=1.2,
                        label=st['label'].split(' (')[0])

        # --- auto-annotate the K=8 failure valley wherever it exists ---
        # (appears at 0/10 dB, absent at 20/25 dB — the annotation itself
        #  visually tells the "valley vanishes with SNR" story)
        m8, _ = _series(stats, 8, snr)
        for vi in _interior_valleys(m8):
            ax.annotate('Valley', xy=(_X_RATIO[vi], m8[vi]),
                        xytext=(-12, 26), textcoords='offset points',
                        fontsize=8.5, fontweight='bold', ha='right',
                        color=_K_STYLE[8]['color'],
                        arrowprops=dict(arrowstyle='->',
                                        color=_K_STYLE[8]['color'], lw=1.1))

        ax.set_title(f'SNR = {int(snr)} dB', fontsize=13, fontweight='bold')
        ax.set_xlabel(r'$\gamma = D_a / D_{\mathrm{tot}}$')
        ax.set_ylabel('Sum-Rate (bps/Hz)')
        ax.set_xticks(_X_RATIO)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    fig.suptitle('Resource-Allocation Sweep across User Loads and SNRs',
                 fontsize=15, fontweight='bold', y=1.00)
    fig.text(0.5, -0.01,
             'Note: y-axis scaled independently per panel; '
             '95% CI is smaller than the marker size.',
             ha='center', fontsize=9, style='italic', color='#555')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        p = os.path.join(RESULTS_DIR, f"fig_phase_transition_multiSNR.{ext}")
        plt.savefig(p, format=ext, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plot] -> fig_phase_transition_multiSNR.{{pdf,png}}")


if __name__ == "__main__":
    main()