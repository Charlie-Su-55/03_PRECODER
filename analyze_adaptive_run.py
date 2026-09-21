#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs")
    parser.add_argument("--exp", required=True)
    args = parser.parse_args()

    metrics_path = (
        Path(args.root)
        / "logs"
        / args.exp
        / "metrics.jsonl"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # allocations = [
    #     (16, 0),
    #     (12, 32),
    #     (8, 64),
    #     (4, 96),
    #     (0, 128),
    # ]
    allocations = [
        (32,0), (24,64), (16,128), (8,192), (0,256),
    ]

    steps = [record["step"] for record in records]

    # =========================================================
    # 1. Find the best checkpoint according to 25-dB average
    # =========================================================
    def average_25db(record):
        return np.mean([
            record["table"][f"alloc{i}_snr25"]
            for i in range(len(allocations))
        ])

    best_25_record = max(records, key=average_25db)

    print("=" * 70)
    print(f"Number of validation records: {len(records)}")
    print(f"Final validation step: {records[-1]['step']}")
    print(
        f"Best mean-all step: "
        f"{max(records, key=lambda x: x['selection_metric'])['step']}"
    )
    print(
        f"Best 25-dB-average step: {best_25_record['step']}, "
        f"value={average_25db(best_25_record):.4f} bps/Hz"
    )

    print("\nBest per-allocation 25-dB values:")
    for i, allocation in enumerate(allocations):
        key = f"alloc{i}_snr25"
        best_record = max(records, key=lambda x: x["table"][key])
        print(
            f"  {allocation}: "
            f"{best_record['table'][key]:.4f} bps/Hz "
            f"at step {best_record['step']}"
        )

    print("\nFinal 25-dB values:")
    final_record = records[-1]
    for i, allocation in enumerate(allocations):
        print(
            f"  {allocation}: "
            f"{final_record['table'][f'alloc{i}_snr25']:.4f}"
        )

    # =========================================================
    # 2. Convergence curves at 25 dB
    # =========================================================
    plt.figure(figsize=(7.2, 4.8))

    for i, allocation in enumerate(allocations):
        values = [
            record["table"][f"alloc{i}_snr25"]
            for record in records
        ]
        plt.plot(steps, values, label=str(allocation))

    plt.xlabel("Training step")
    plt.ylabel("Sum rate at 25 dB (bps/Hz)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{args.exp}_convergence_25db.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # =========================================================
    # 3. Final allocation sweep at 25 dB
    # =========================================================
    aircomp_fraction = [0, 25, 50, 75, 100]
    final_values = [
        final_record["table"][f"alloc{i}_snr25"]
        for i in range(len(allocations))
    ]

    plt.figure(figsize=(6.4, 4.5))
    plt.plot(aircomp_fraction, final_values, marker="o")
    plt.xlabel("AirComp resource fraction (%)")
    plt.ylabel("Sum rate at 25 dB (bps/Hz)")
    plt.xticks(aircomp_fraction)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{args.exp}_allocation_25db.pdf",
        bbox_inches="tight",
    )
    plt.close()

    print("\nSaved:")
    print(f"  {args.exp}_convergence_25db.pdf")
    print(f"  {args.exp}_allocation_25db.pdf")


if __name__ == "__main__":
    main()