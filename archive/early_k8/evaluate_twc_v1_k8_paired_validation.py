#!/usr/bin/env python3
"""Paired CPU-only validation for the completed K8 Stage-B checkpoints.

Uses a fresh, deterministic validation-role channel set and common pilot/data
noise tensors across all arms, seeds, Hybrid group counts, and energy splits.
The test role is never sampled. This script performs inference only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import torch

from configs import twc_v1_k8_codec_warmup_config as cfg

ROOT = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "runs/twc_v1_k8_codec_warmup/paired_validation_cpu_20261106"


def load_local(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


system = load_local("utils/twc_v1_k8_system.py", "_twc_v1_k8_pair_eval_system")
data_module = load_local("utils/twc_v1_k8_replayable_data.py", "_twc_v1_k8_pair_eval_data")
protocol = system.protocol
TWCv1K8ReplayableData = data_module.TWCv1K8ReplayableData


def fixed_noise(batch: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn((batch, 32, 256, 2), generator=gen, dtype=torch.float32)
    return torch.view_as_complex(z.contiguous()).to(device)


def per_sample_ul_nmse(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    err = (estimate - truth).abs().square().sum(dim=(-1, -2))
    power = truth.abs().square().sum(dim=(-1, -2)).clamp_min(1e-20)
    return (err / power).mean(dim=-1)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    if args.batches < 2 or args.batch_size < 1:
        raise ValueError("Use at least two validation batches and a positive batch size")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite validation evidence: {output_dir}")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    batch_sizes = {k: args.batch_size for k in
                   ("ce_train", "codec_train", "ce_calibration", "validation", "test")}
    # Generate the fixed role-addressed validation set on CPU. Only inference
    # and noise tensors move to the explicitly selected evaluation device.
    data = TWCv1K8ReplayableData(base_seed=cfg.BASE_DATA_SEED,
        batch_sizes=batch_sizes, device="cpu", cdl_model=cfg.CDL_MODEL,
        speed=cfg.SPEED_MPS, seed_namespace=cfg.SEED_NAMESPACE)
    chain = system.K8FeedbackChain.from_artifacts(ROOT / cfg.CE_CHECKPOINT,
        ROOT / cfg.CE_CALIBRATION, device=device)
    chain.eval()

    run_specs = [
        ("fdma", 20261106, ROOT / "runs/twc_v1_k8_codec_warmup/fdma/seed_20261106_jointv1"),
        ("fdma", 20261107, ROOT / "runs/twc_v1_k8_codec_warmup/fdma/seed_20261107_jointv1"),
        ("hybrid", 20261106, ROOT / "runs/twc_v1_k8_codec_warmup/hybrid/seed_20261106_jointv1"),
        ("hybrid", 20261107, ROOT / "runs/twc_v1_k8_codec_warmup/hybrid/seed_20261107_jointv1"),
    ]
    checkpoints = {}
    for arm, seed, run_dir in run_specs:
        path = run_dir / "checkpoint_latest.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("arm") != arm or payload.get("seed") != seed or payload.get("step") != 10000:
            raise ValueError(f"Unexpected checkpoint metadata: {path}")
        chain.codec_model.load_state_dict(payload["codec_model"], strict=True)
        digest = hash_file(path)
        checkpoints[(arm, seed)] = {"path": str(path), "sha256": digest}
    print(f"Loaded 4 verified 10k checkpoints; device={device}; test role will not be sampled.", flush=True)

    ul_rows, dl_rows, fingerprints = [], [], []
    ul_fingerprints: dict[tuple, bytes] = {}
    for bi in range(args.batches):
        sample = data.sample("validation", bi)
        Hdl, Hul = sample["H_dl"].to(device), sample["H_ul"].to(device)
        fingerprints.append({"batch": bi, "sample_fingerprint": sample["sample_fingerprint"]})
        B = Hdl.shape[0]
        for gi, gamma in enumerate(cfg.GAMMAS):
            for ei, ep in enumerate(cfg.PILOT_ENERGIES):
                key = 50_000_000 + bi * 100_000 + gi * 1000 + ei * 10
                pilot_noise = fixed_noise(B, key, device)
                pilot_y = system.link.transmit_pilots(Hul, ep, gamma, pilot_noise)
                hls = system.link.estimate_ul_ls_interpolated(pilot_y, ep)
                pilots = chain._known_pilots(B, device)
                hunet = chain.ce_model(pilot_y, pilots, hls,
                    pilot_energy_per_ue=ep,
                    noise_variance=protocol.feedback_noise_variance(gamma))
                ls_values = per_sample_ul_nmse(hls, Hul)
                unet_values = per_sample_ul_nmse(hunet, Hul)
                for si in range(B):
                    ul_rows.append({"batch": bi, "sample_in_batch": si,
                        "sample_fingerprint": sample["sample_fingerprint"],
                        "gamma_db": gamma, "pilot_energy_per_ue": ep,
                        "pilot_RE": 64, "estimator": "LS_interpolated",
                        "ul_nmse": float(ls_values[si]),
                        "ul_nmse_db": float(10 * torch.log10(ls_values[si].clamp_min(1e-20)))})
                    ul_rows.append({"batch": bi, "sample_in_batch": si,
                        "sample_fingerprint": sample["sample_fingerprint"],
                        "gamma_db": gamma, "pilot_energy_per_ue": ep,
                        "pilot_RE": 64, "estimator": "trained_U_Net",
                        "ul_nmse": float(unet_values[si]),
                        "ul_nmse_db": float(10 * torch.log10(unet_values[si].clamp_min(1e-20)))})
                ul_fingerprints[(bi, gamma, ep)] = hunet.detach().contiguous().cpu().numpy().tobytes()

        gamma_ep = 16.0
        for gi, gamma in enumerate(cfg.GAMMAS):
            pilot_noise = fixed_noise(B, 50_000_000 + bi * 100_000 + gi * 1000 + 10, device)
            data_noise = fixed_noise(B, 60_000_000 + bi * 100_000 + gi * 1000, device)
            for arm, seed, _ in run_specs:
                meta = checkpoints[(arm, seed)]
                payload = torch.load(meta["path"], map_location="cpu", weights_only=False)
                chain.codec_model.load_state_dict(payload["codec_model"], strict=True)
                payload = None
                if arm == "fdma":
                    configs = [(1, "fdma", None)]
                else:
                    configs = []
                    for groups in (1, 2, 4):
                        for ratio in ((1.0, 3.0), (1.0, 1.0)):
                            configs.append((groups, f"hybrid_{ratio[0]:g}_{ratio[1]:g}", ratio))
                for groups, layout_name, ratio in configs:
                    if arm == "fdma":
                        layout, split = protocol.FDMALayout(), None
                    else:
                        layout = protocol.default_hybrid_layout(groups)
                        data_energy = protocol.TOTAL_FRAME_ENERGY_PER_USER - gamma_ep
                        split = protocol.EnergySplit(gamma_ep,
                            data_energy * ratio[0] / sum(ratio),
                            data_energy * ratio[1] / sum(ratio))
                    output = chain(Hdl, Hul, feedback_gamma_db=gamma,
                        pilot_energy_per_ue=gamma_ep, pilot_standard_noise=pilot_noise,
                        data_standard_noise=data_noise, layout=layout, energy_split=split)
                    metrics = chain.metrics(output, Hdl)
                    ul_bytes = output["H_ul_est"].detach().contiguous().cpu().numpy().tobytes()
                    expected_ul = ul_fingerprints[(bi, gamma, gamma_ep)]
                    if ul_bytes != expected_ul:
                        raise AssertionError(f"UL estimate differs across layouts at batch={bi}, gamma={gamma}")
                    nmse = metrics["dl_nmse_per_user"].mean(dim=-1)
                    rate = metrics["sum_rate_bps_hz"]
                    for si in range(B):
                        dl_rows.append({"batch": bi, "sample_in_batch": si,
                            "sample_fingerprint": sample["sample_fingerprint"],
                            "arm": arm, "seed": seed, "checkpoint_step": 10000,
                            "group_count": groups, "energy_policy": layout_name,
                            "gamma_db": gamma, "pilot_energy_per_ue": gamma_ep,
                            "key_energy_per_ue": 0.0 if split is None else split.key,
                            "shared_energy_per_ue": 48.0 if split is None else split.shared,
                            "loss_stage": "NMSE_warmup", "learning_rate": 2e-4,
                            "loss_weight_nmse": 1.0, "loss_weight_rate": 0.0,
                            "dl_nmse": float(nmse[si]),
                            "downlink_sum_rate_bps_hz": float(rate[si])})
        print(f"Completed validation batch {bi + 1}/{args.batches} ({B} samples).", flush=True)

    out = output_dir
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "ul_nmse_paired_validation.csv", ul_rows)
    write_csv(out / "dl_metrics_paired_validation.csv", dl_rows)
    # Aggregate sample-level rows while preserving condition-level point estimates.
    summaries = []
    for arm, seed, _ in run_specs:
        configs = [(1, "fdma")] if arm == "fdma" else [
            (g, f"hybrid_{r[0]:g}_{r[1]:g}") for g in (1, 2, 4) for r in ((1., 3.), (1., 1.))]
        for groups, policy in configs:
            for gamma in cfg.GAMMAS:
                rows = [r for r in dl_rows if r["arm"] == arm and r["seed"] == seed
                    and r["group_count"] == groups and r["energy_policy"] == policy
                    and r["gamma_db"] == gamma]
                summaries.append({"arm": arm, "seed": seed, "group_count": groups,
                    "energy_policy": policy, "gamma_db": gamma,
                    "n_paired_samples": len(rows),
                    "dl_nmse_mean": sum(r["dl_nmse"] for r in rows) / len(rows),
                    "downlink_sum_rate_mean_bps_hz": sum(r["downlink_sum_rate_bps_hz"] for r in rows) / len(rows)})
    write_csv(out / "dl_metrics_summary.csv", summaries)
    sample_metrics: dict[tuple, dict[int, tuple[float, float]]] = {}
    for row in dl_rows:
        key = (row["arm"], row["seed"], row["group_count"], row["energy_policy"],
               row["batch"], row["sample_in_batch"])
        sample_metrics.setdefault(key, {})[int(float(row["gamma_db"]))] = (
            row["dl_nmse"], row["downlink_sum_rate_bps_hz"])
    six_snr = {key: (sum(v[0] for v in values.values()) / len(values),
                     sum(v[1] for v in values.values()) / len(values))
               for key, values in sample_metrics.items()}
    rng = random.Random(620261106)
    paired_rows = []
    for seed in (20261106, 20261107):
        fdma = {key[-2:]: value for key, value in six_snr.items()
                if key[0] == "fdma" and key[1] == seed}
        for groups in (1, 2, 4):
            for ratio_name in ("hybrid_1_3", "hybrid_1_1"):
                hybrid = {key[-2:]: value for key, value in six_snr.items()
                          if key[0] == "hybrid" and key[1] == seed
                          and key[2] == groups and key[3] == ratio_name}
                sample_ids = sorted(set(fdma) & set(hybrid))
                if len(sample_ids) != args.batches * args.batch_size:
                    raise AssertionError("FDMA/Hybrid validation sample pairing is incomplete")
                for metric_index, metric in enumerate(("dl_nmse", "downlink_sum_rate_bps_hz")):
                    deltas = [hybrid[sid][metric_index] - fdma[sid][metric_index]
                              for sid in sample_ids]
                    boot = []
                    for _ in range(10000):
                        boot.append(sum(deltas[rng.randrange(len(deltas))]
                                        for _ in range(len(deltas))) / len(deltas))
                    boot.sort()
                    paired_rows.append({"seed": seed, "group_count": groups,
                        "hybrid_energy_policy": ratio_name, "metric": metric,
                        "n_paired_samples": len(sample_ids),
                        "fdma_mean_six_snr": sum(fdma[sid][metric_index] for sid in sample_ids) / len(sample_ids),
                        "hybrid_mean_six_snr": sum(hybrid[sid][metric_index] for sid in sample_ids) / len(sample_ids),
                        "paired_delta_hybrid_minus_fdma": sum(deltas) / len(deltas),
                        "sample_bootstrap_95_low": boot[int(0.025 * len(boot))],
                        "sample_bootstrap_95_high": boot[min(int(0.975 * len(boot)), len(boot) - 1)],
                        "bootstrap_unit": "sample; each sample averaged over six SNRs"})
    write_csv(out / "paired_delta_summary.csv", paired_rows)
    ul_summary = []
    for gamma in cfg.GAMMAS:
        for ep in cfg.PILOT_ENERGIES:
            for estimator in ("LS_interpolated", "trained_U_Net"):
                rows = [r for r in ul_rows if r["gamma_db"] == gamma
                    and r["pilot_energy_per_ue"] == ep and r["estimator"] == estimator]
                mean_nmse = sum(r["ul_nmse"] for r in rows) / len(rows)
                ul_summary.append({"gamma_db": gamma, "pilot_energy_per_ue": ep,
                    "pilot_RE": 64, "estimator": estimator, "n_samples": len(rows),
                    "ul_nmse_mean": mean_nmse,
                    "ul_nmse_db_of_mean": 10 * torch.log10(torch.tensor(mean_nmse)).item()})
    write_csv(out / "ul_nmse_summary.csv", ul_summary)
    manifest = {"status": "completed", "evaluation": "validation_only_paired_cpu_inference",
        "test_used": False, "optimizer_updates": 0, "device": str(device),
        "channel_generation_device": "cpu",
        "data_protocol": data.protocol, "batches": args.batches,
        "batch_size": args.batch_size, "samples": args.batches * args.batch_size,
        "sample_fingerprints": fingerprints,
        "common_noise": True, "paired_ul_estimates_bitwise_equal_across_layouts": True,
        "pilot_energies_per_ue": list(cfg.PILOT_ENERGIES), "gamma_db": list(cfg.GAMMAS),
        "downlink_pilot_energy_per_ue": 16.0,
        "hybrid_energy_ratios": [[1, 3], [1, 1]],
        "checkpoints": {f"{arm}_seed{seed}": value for (arm, seed), value in checkpoints.items()},
        "interpretation": "Validation-only exploratory comparison. No independent performance claim; all checkpoints are Stage-B NMSE warmups."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status=completed device=cpu samples={args.batches * args.batch_size} test_used=False optimizer_updates=0")
    print(f"data_id={data.protocol['data_id']} output={out}")
    print("UL NMSE (mean over validation samples):")
    for row in ul_summary:
        print(f"  gamma={row['gamma_db']:>2} dB Ep={row['pilot_energy_per_ue']:>2} "
              f"{row['estimator']}: NMSE={row['ul_nmse_mean']:.6f} "
              f"({row['ul_nmse_db_of_mean']:.2f} dB)")
    print("DL paired validation summary by seed/arm/G/energy policy saved to dl_metrics_summary.csv")


if __name__ == "__main__":
    main()
