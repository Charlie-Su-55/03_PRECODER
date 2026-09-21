#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math


MCS = {
    "QPSK": 2,
    "16QAM": 4,
    "64QAM": 6,
}


def type2_bits_per_ue(Nt, L, n_subbands, amp_bits, phase_bits, oversample=4):
    n_beams = Nt * oversample
    beam_bits = math.ceil(math.log2(math.comb(n_beams, L)))
    amp_total = L * n_subbands * amp_bits - amp_bits
    phase_total = L * n_subbands * phase_bits - phase_bits
    total = beam_bits + amp_total + phase_total
    return beam_bits, amp_total, phase_total, total


def channel_uses_per_ue(info_bits, coderate, bits_per_symbol):
    coded_bits = math.ceil(info_bits / coderate)
    symbols = math.ceil(coded_bits / bits_per_symbol)
    return coded_bits, symbols


def main():
    parser = argparse.ArgumentParser(description="Audit Type-II feedback budget under fixed complex channel uses.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--oversample", type=int, default=4)
    args = parser.parse_args()

    source_configs = [
        ("Type-I L1", 1, 4, 3, 3),
        ("Type-II L2", 2, 4, 3, 3),
        ("Type-II L4", 4, 4, 3, 3),
        ("Type-II L4-SB2", 4, 2, 3, 3),
        ("eType-II L4-SB2", 4, 2, 4, 4),
    ]
    coderates = [1.0, 0.75, 2.0 / 3.0, 0.5]

    print("=" * 112)
    print(f"Type-II digital budget audit: K={args.k}, Nt={args.nt}, Dtot={args.budget} complex uses")
    print("=" * 112)
    print(
        f"{'source':<20} {'bits/UE':>8} {'coding':>7} {'mod':>7} "
        f"{'coded/UE':>10} {'sym/UE':>8} {'total sym':>10} {'fit':>6}"
    )
    print("-" * 112)

    for label, L, n_sb, amp_bits, phase_bits in source_configs:
        _, _, _, info_bits = type2_bits_per_ue(
            args.nt, L, n_sb, amp_bits, phase_bits, args.oversample
        )

        for coderate in coderates:
            for mod_name, bps in MCS.items():
                coded_bits, symbols_per_ue = channel_uses_per_ue(info_bits, coderate, bps)
                total_symbols = args.k * symbols_per_ue
                fit = "YES" if total_symbols <= args.budget else "NO"

                print(
                    f"{label:<20} {info_bits:>8} {coderate:>7.3f} {mod_name:>7} "
                    f"{coded_bits:>10} {symbols_per_ue:>8} {total_symbols:>10} {fit:>6}"
                )

        print("-" * 112)

    print("\nNotes:")
    print("1. 'total sym' is the aggregate orthogonal digital feedback cost over all K UEs.")
    print("2. A fair configuration must satisfy total sym <= Dtot.")
    print("3. Unused symbols are allowed; exceeding Dtot is not.")
    print("4. This audit only checks the resource budget; it does not evaluate rate yet.")


if __name__ == "__main__":
    main()