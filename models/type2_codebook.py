#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/type2_codebook.py
============================
5G NR Type II Codebook Baseline (3GPP TS 38.214 Rel-15 风格)

修复说明 (本版):
  - ZF 正则项区分 Perfect H 和 quantized H_hat 两种模式
    Perfect: 1e-4 (保数值稳定)
    Quantized: K/SNR + 0.05 (考虑量化噪声)
  - 诊断函数 mask 使用 expand 避免 broadcast 不兼容
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np


def _amplitude_codebook(n_bits):
    if n_bits == 3:
        levels = torch.tensor([
            0.0, 1.0/8, 1.0/(4*np.sqrt(2)), 1.0/4,
            1.0/(2*np.sqrt(2)), 1.0/2, 1.0/np.sqrt(2), 1.0
        ])
    elif n_bits == 4:
        N = 2 ** n_bits
        levels = torch.tensor([0.0] + [
            np.sqrt(1.0 / 2 ** ((N - 1 - i))) for i in range(N - 1)
        ])
    else:
        levels = torch.linspace(0, 1, 2 ** n_bits)
    return levels


def quantize_amplitude(amp_normalized, n_bits):
    levels = _amplitude_codebook(n_bits).to(amp_normalized.device)
    diff = (amp_normalized.unsqueeze(-1) - levels).abs()
    idx = diff.argmin(dim=-1)
    return levels[idx]


def quantize_phase(phase, n_bits):
    n_levels = 2 ** n_bits
    step = 2 * np.pi / n_levels
    phase_wrapped = torch.remainder(phase + np.pi, 2 * np.pi)
    idx = torch.round(phase_wrapped / step).long() % n_levels
    return idx.to(phase.dtype) * step - np.pi


def generate_oversampled_dft_codebook(Nt, oversample=4):
    N_beams = Nt * oversample
    n_idx = torch.arange(Nt).float()
    i_idx = torch.arange(N_beams).float()
    phase_matrix = 2 * np.pi * torch.outer(n_idx, i_idx) / N_beams
    B = torch.exp(1j * phase_matrix) / np.sqrt(Nt)
    return B.to(torch.cfloat)


class Type2CodebookBaseline(nn.Module):
    def __init__(self, cfg, L=4, amp_bits=3, phase_bits=3, oversample=4,
                 n_subbands=None, verbose=False):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K_USERS
        self.Nt = cfg.ANTENNAS
        self.Nsc = cfg.SUBCARRIERS
        self.L = L
        self.amp_bits = amp_bits
        self.phase_bits = phase_bits
        self.oversample = oversample
        self.n_subbands = n_subbands if n_subbands is not None else \
                          getattr(cfg, 'NUM_SUBBANDS', 8)
        assert self.Nsc % self.n_subbands == 0
        self.sb_size = self.Nsc // self.n_subbands
        self.verbose = verbose
        
        codebook = generate_oversampled_dft_codebook(self.Nt, self.oversample)
        self.register_buffer('codebook', codebook)
        self._compute_feedback_overhead()
    
    def _compute_feedback_overhead(self):
        from math import comb, log2
        N_beams = self.Nt * self.oversample
        bits_beam_sel = log2(comb(N_beams, self.L))
        bits_amp = self.L * self.n_subbands * self.amp_bits - self.amp_bits
        bits_phase = self.L * self.n_subbands * self.phase_bits - self.phase_bits
        self.bits_per_ue = bits_beam_sel + bits_amp + bits_phase
        if self.verbose:
            print(f"[Type II] L={self.L}, oversample={self.oversample}, "
                  f"n_subbands={self.n_subbands}")
            print(f"[Type II] Beam selection: {bits_beam_sel:.1f} bit | "
                  f"Amp: {bits_amp} bit | Phase: {bits_phase} bit")
            print(f"[Type II] Total per UE: {self.bits_per_ue:.1f} bit")
    
    def encode_one_ue(self, h_eq):
        B = h_eq.shape[0]
        h_sb = h_eq.reshape(B, self.Nt, self.n_subbands, self.sb_size).mean(dim=-1)

        # ------------------------------------------------------------------
        # Greedy residual beam selection.
        # The oversampled DFT codebook is not orthogonal, so simply taking
        # the top-L projections can repeatedly select highly correlated beams.
        # ------------------------------------------------------------------
        residual = h_sb.clone()
        selected_idx = []

        codebook_batched = self.codebook.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)

        for _ in range(self.L):
            proj = torch.einsum("nb,Bns->Bbs", self.codebook.conj(), residual)
            score = proj.abs().square().sum(dim=-1)

            if selected_idx:
                previous = torch.stack(selected_idx, dim=1)
                score.scatter_(1, previous, float("-inf"))

            idx = score.argmax(dim=1)
            selected_idx.append(idx)

            beam = codebook_batched[
                torch.arange(B, device=h_eq.device), idx
            ]  # [B, Nt]

            coeff = torch.einsum("Bn,Bns->Bs", beam.conj(), residual)
            residual = residual - beam.unsqueeze(-1) * coeff.unsqueeze(1)

        top_L_idx = torch.stack(selected_idx, dim=1)  # [B, L]

        # ------------------------------------------------------------------
        # Joint LS coefficient fitting after beam selection.
        # This correctly accounts for correlation among oversampled beams.
        # ------------------------------------------------------------------
        idx_expand = top_L_idx.unsqueeze(-1).expand(B, self.L, self.Nt)
        selected_beams = codebook_batched.gather(1, idx_expand)
        selected_beams = selected_beams.transpose(1, 2).contiguous()  # [B, Nt, L]

        gram = selected_beams.conj().transpose(1, 2) @ selected_beams
        rhs = torch.einsum("Bnl,Bns->Bls", selected_beams.conj(), h_sb)

        eye = torch.eye(self.L, dtype=gram.dtype, device=gram.device).unsqueeze(0)
        coeffs = torch.linalg.solve(gram + 1e-4 * eye, rhs)

        # ------------------------------------------------------------------
        # Quantize relative amplitude and phase.
        # ------------------------------------------------------------------
        amp = coeffs.abs()
        max_amp = amp.amax(dim=(1, 2), keepdim=True).clamp_min(1e-10)
        amp_norm = amp / max_amp
        amp_q = quantize_amplitude(amp_norm, self.amp_bits) * max_amp

        phase_q = quantize_phase(coeffs.angle(), self.phase_bits)
        coeffs_q = amp_q * torch.exp(1j * phase_q)

        return top_L_idx, coeffs_q.to(torch.cfloat)
    
    def decode_one_ue(self, top_L_idx, coeffs_q):
        B = top_L_idx.shape[0]
        idx_expand = top_L_idx.unsqueeze(1).expand(B, self.Nt, self.L)
        beams = self.codebook.unsqueeze(0).expand(B, -1, -1)
        selected_beams = beams.gather(2, idx_expand)
        H_hat_sb = torch.einsum('Bnl,Bls->Bns', selected_beams, coeffs_q)
        H_hat = H_hat_sb.repeat_interleave(self.sb_size, dim=-1)
        return H_hat
    
    def zf_precoder(self, H_hat_all, snr_db=15, quant_noise=True):
        """
        Args:
            quant_noise: True 用于 Type II quantized H_hat,
                        False 用于 Perfect H (上界测试)
        """
        B, K, Nt, Nsc = H_hat_all.shape
        device = H_hat_all.device
        H = H_hat_all.permute(0, 3, 1, 2)
        norm_per_sc = torch.norm(H, dim=(-2, -1), keepdim=True) + 1e-8
        H_n = H / norm_per_sc
        H_H = H_n.conj().transpose(-2, -1)
        H_gram = H_n @ H_H
        
        snr_linear = 10 ** (snr_db / 10)
        if quant_noise:
            # 量化 NMSE 经验值随 L 和 amp_bits 变化
            # L=4, 3-bit: NMSE ~ 0.05-0.1
            # L=4, 4-bit: NMSE ~ 0.02-0.05
            # 这里用 0.02 作为更精细的下限, 大 SNR 下不再被这项拖累
            quant_nmse = 0.02 if self.amp_bits >= 4 else 0.05
            reg_value = self.K / snr_linear + quant_nmse
        else:
            reg_value = max(1e-4, self.K / snr_linear * 0.01)
        
        reg = reg_value * torch.eye(K, device=device, dtype=torch.cfloat)
        H_gram_reg = H_gram + reg.unsqueeze(0).unsqueeze(0)
        W_per_sc = H_H @ torch.linalg.inv(H_gram_reg)
        W = W_per_sc.permute(0, 2, 3, 1)
        frob = (W.abs() ** 2).sum(dim=(1, 2), keepdim=True)
        W = W * torch.sqrt(self.cfg.TOTAL_POWER / (frob + 1e-9))
        return W
    
    @torch.no_grad()
    def forward(self, H_dl, snr_db=15):
        B, K, Nt, Nsc = H_dl.shape
        H_eq = H_dl.conj()
        H_hat_list = []
        for k in range(K):
            top_L_idx, coeffs_q = self.encode_one_ue(H_eq[:, k])
            H_hat_k = self.decode_one_ue(top_L_idx, coeffs_q)
            H_hat_list.append(H_hat_k)
        H_hat_all = torch.stack(H_hat_list, dim=1)
        # quant_noise=True 因为这是 Type II 重建的 H_hat
        W = self.zf_precoder(H_hat_all, snr_db=snr_db, quant_noise=True)
        return W.to(torch.cfloat)
    
    @torch.no_grad()
    def diagnose_zf_conditioning(self, H_dl):
        """诊断 H_hat 的多用户方向坍塌程度"""
        H_eq = H_dl.conj()
        H_hat_list = []
        for k in range(self.K):
            top_L_idx, coeffs_q = self.encode_one_ue(H_eq[:, k])
            H_hat_k = self.decode_one_ue(top_L_idx, coeffs_q)
            H_hat_list.append(H_hat_k)
        H_hat_all = torch.stack(H_hat_list, dim=1)
        
        K = self.K
        device = H_hat_all.device
        # 修复: mask expand 到完整 shape
        eye_mask = torch.eye(K, dtype=torch.bool, device=device)
        off_mask = ~eye_mask  # [K, K]
        
        # Type II H_hat 的方向相关
        H = H_hat_all.permute(0, 3, 1, 2)  # [B, Nsc, K, Nt]
        H_norm = H / (H.norm(dim=-1, keepdim=True) + 1e-9)
        cross = H_norm @ H_norm.conj().transpose(-2, -1)  # [B, Nsc, K, K]
        # 展开 mask 到 [B, Nsc, K, K] 然后用 boolean indexing
        off_diag = cross[:, :, off_mask].abs().mean()
        
        # Perfect H 的方向相关
        H_true = H_dl.conj().permute(0, 3, 1, 2)
        H_true_n = H_true / (H_true.norm(dim=-1, keepdim=True) + 1e-9)
        cross_true = H_true_n @ H_true_n.conj().transpose(-2, -1)
        off_diag_true = cross_true[:, :, off_mask].abs().mean()
        
        # Gram 条件数
        gram_hat = H @ H.conj().transpose(-2, -1)
        gram_true = H_true @ H_true.conj().transpose(-2, -1)
        cond_hat = torch.linalg.cond(gram_hat).mean()
        cond_true = torch.linalg.cond(gram_true).mean()
        
        print(f"  Off-diag user correlation (Type II H_hat): {off_diag.item():.4f}")
        print(f"  Off-diag user correlation (Perfect H):     {off_diag_true.item():.4f}")
        print(f"  Gram condition number (Type II): {cond_hat.item():.2f}")
        print(f"  Gram condition number (Perfect): {cond_true.item():.2f}")
        if off_diag.item() > 0.5:
            print(f"  ⚠️  H_hat 多用户方向严重坍塌, MMSE 正则会有效")
        elif off_diag.item() > 0.3:
            print(f"  ⚠️  H_hat 多用户方向部分坍塌")
        else:
            print(f"  ✅ H_hat 多用户方向分离良好")


# ================================================================
# 一键评估脚本
# ================================================================
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from configs.exp_4ue_128sc import HybridPrecoderConfig as cfg
    from utils import CDLChannelGenerator_Feedback, setup_gpu
    from nouse import SumRateLoss
    
    device = setup_gpu()
    channel_gen = CDLChannelGenerator_Feedback(
        Nt=cfg.ANTENNAS, Nsc=cfg.SUBCARRIERS,
        speed=cfg.SPEED, num_users=cfg.K_USERS
    )
    
    criterion = SumRateLoss()
    snr_list = [0, 5, 10, 15, 20, 25]
    
    # ---------- Step 1: Perfect ZF 上界 ----------
    print("\n" + "=" * 70)
    print("STEP 1: Perfect ZF Upper Bound (quant_noise=False)")
    print("=" * 70)
    
    baseline_test = Type2CodebookBaseline(cfg, L=4, amp_bits=3, phase_bits=3).to(device)
    H_dl, _ = channel_gen.generate_batch_data(64)
    H_dl = H_dl.to(device)
    
    for snr in snr_list:
        # 关键: quant_noise=False, 因为是 Perfect H
        W = baseline_test.zf_precoder(H_dl.conj(), snr_db=snr, quant_noise=False)
        np_v = 10 ** (-snr / 10)
        _, rate = criterion(W, H_dl, np_v)
        print(f"  Perfect ZF @ {snr:>3}dB: {rate:.3f} bps/Hz")
    
    # ---------- Step 2: 诊断 ----------
    print("\n" + "=" * 70)
    print("STEP 2: Diagnose ZF Conditioning (L=4 Type II)")
    print("=" * 70)
    baseline_test.diagnose_zf_conditioning(H_dl)
    
    # ---------- Step 3: 4 个 Type II 配置 ----------
    print("\n" + "=" * 70)
    print("STEP 3: Full Type II Baseline Evaluation")
    print("=" * 70)
    
    configs_to_test = [
        ("Type I (L=1)",        1, 3, 3),
        ("Type II Rel-15 L=2",  2, 3, 3),
        ("Type II Rel-15 L=4",  4, 3, 3),
        ("eType II Rel-16 L=4", 4, 4, 4),
    ]
    
    all_results = {}
    
    for label, L, ab, pb in configs_to_test:
        print(f"\n{'#' * 60}\n# {label}\n{'#' * 60}")
        baseline = Type2CodebookBaseline(
            cfg, L=L, amp_bits=ab, phase_bits=pb,
            n_subbands=cfg.NUM_SUBBANDS, verbose=True
        ).to(device)
        
        results = {snr: [] for snr in snr_list}
        with torch.no_grad():
            for _ in range(8):
                H_dl_b, _ = channel_gen.generate_batch_data(64)
                H_dl_b = H_dl_b.to(device)
                for snr in snr_list:
                    W = baseline(H_dl_b, snr_db=snr)
                    np_v = 10 ** (-snr / 10)
                    _, rate = criterion(W, H_dl_b, np_v)
                    results[snr].append(rate)
        
        avg = {s: float(np.mean(v)) for s, v in results.items()}
        all_results[label] = avg
        print(f"\n  Results @ different SNRs:")
        for s, r in avg.items():
            print(f"    SNR={s:>3} dB → {r:.3f} bps/Hz")
    
    # ---------- Step 4: 保存 ----------
    os.makedirs("results/type2", exist_ok=True)
    csv_path = "results/type2/type2_results.csv"
    with open(csv_path, 'w') as f:
        f.write("Method," + ",".join([f"{s}dB" for s in snr_list]) + "\n")
        for label, res in all_results.items():
            row = [label] + [f"{res[s]:.3f}" for s in snr_list]
            f.write(",".join(row) + "\n")
    
    print("\n" + "=" * 80)
    print(f"{'Method':<25}" + "".join([f"{s:>9}dB" for s in snr_list]))
    print("=" * 80)
    for label, res in all_results.items():
        rates_str = "".join([f"{res[s]:>11.3f}" for s in snr_list])
        print(f"{label:<25}{rates_str}")
    print("=" * 80)
    print(f"\n[Done] Results saved to: {csv_path}")