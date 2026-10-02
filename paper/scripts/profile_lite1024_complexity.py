#!/usr/bin/env python3
"""Table-V UE-only profiling and a separate, CSV-only K4 rate-loss audit.

No weights are loaded; no channels, rates, training or latency are evaluated.
--profile needs the repository's model dependencies (including sionna.phy).
--tradeoff needs only the standard library. Existing different outputs fail
closed; identical generated CSVs are accepted without rewriting them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "paper/data/k4_pure_as_lite1024/formal_20261002T124602_541062Z"
FORMAL_FILES = (
    "summary_k4_pure_as_lite1024.csv",
    "paired_deltas_k4_pure_as_lite1024.csv",
    "parameters_k4_pure_as_lite1024.csv",
    "manifest.json",
)
FB_SNRS = (0, 5, 10, 15, 20, 25)
ALLOCATIONS = ((4, 96), (16, 0), (0, 128))
TABLEV_PARAMS_M = "10.756"
TABLEV_MACS_M = {(4, 96): "345.1", (16, 0): "174.1", (0, 128): "171.1"}
MODEL_SOURCES = (ROOT / "models/adaptive_hybrid.py", ROOT / "models/adaptive_hybrid_lite.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha256_lf(path):
    # Git's Windows checkout converts manifest line endings. Use explicitly
    # labelled LF-normalized provenance in portable CSVs; keep raw byte hashes
    # for before/after file-integrity checks.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def number(row, key):
    value = float(row[key])
    require(math.isfinite(value), f"Non-finite {key}: {row}")
    return value


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unique_index(rows, key_fn):
    result = {}
    for row in rows:
        key = key_fn(row)
        require(key not in result, f"Duplicate key: {key}")
        result[key] = row
    return result


def check_role(row):
    require(row.get("preflight") in (False, "False", "false"), "Preflight is not formal data")
    require(row.get("dataset_role") == "independent_held_out_test", "Wrong dataset role")


def check_ci(row, prefix):
    mean, se, low, high = (number(row, prefix + suffix) for suffix in
                           ("mean", "se", "ci95_low", "ci95_high"))
    require(se >= 0 and low <= mean <= high, f"Invalid CI: {row}")
    require(math.isclose(low, mean - 1.96 * se, abs_tol=1e-10, rel_tol=1e-10)
            and math.isclose(high, mean + 1.96 * se, abs_tol=1e-10, rel_tol=1e-10),
            f"CI is not the saved 1.96-SE interval: {row}")


def load_tradeoff(formal_dir=FORMAL):
    """Read exactly four formal artifacts; preserve signed paired statistics."""
    formal_dir = Path(formal_dir)
    require(formal_dir.name == FORMAL.name, "Use the specified formal run, never preflight")
    paths = {name: formal_dir / name for name in FORMAL_FILES}
    before = {name: sha256(path) for name, path in paths.items()}
    source_hashes = {name: text_sha256_lf(path) for name, path in paths.items()}
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    check_role(manifest)
    expected = {
        "status": "completed", "seed": 20261002, "batch_size": 8,
        "num_batches": 200, "num_samples_per_condition": 1600,
        "feedback_snr_db": list(FB_SNRS), "downlink_snr_db": 25,
        "summary_rows": 24, "checkpoint_and_buffers_unchanged": True,
        "scenario": {"k_users": 4, "antennas": 32, "subcarriers": 256,
                     "feedback_budget": 256, "num_subbands": 4},
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, f"Formal manifest mismatch: {key}")
    for method, width in (("original", 2048), ("lite1024", 1024)):
        saved = manifest["methods"][method]
        require(saved["allocation"] == [0, 256] and saved["max_dimensions"] == [64, 256],
                f"Wrong K4 allocation/capacity: {method}")
        require(saved["completion_step"] == 50000, f"Incomplete training: {method}")
        require(saved["structure_widths"] == {
            "fdma_backbone": [width] * 4, "aircomp_backbone": [width] * 4,
            "bs_cross_subband": [2048], "bs_refinement": [512] * 5,
        }, f"Unexpected structure widths: {method}")

    summary = unique_index(csv_rows(paths[FORMAL_FILES[0]]),
                           lambda r: (r["method_key"], number(r, "feedback_snr_db")))
    require(set(summary) == {(m, s) for m in ("original", "lite1024", "swin", "csinet")
                             for s in FB_SNRS}, "Missing/extra summary conditions")
    paired = unique_index(csv_rows(paths[FORMAL_FILES[1]]),
                          lambda r: (r["comparison"], number(r, "feedback_snr_db")))
    require(set(paired) == {(f"lite1024-{m}", s) for m in ("original", "swin")
                            for s in FB_SNRS}, "Missing/extra paired conditions")
    params = unique_index(csv_rows(paths[FORMAL_FILES[2]]), lambda r: r["method_key"])
    require(set(params) == {"original", "lite1024", "swin", "csinet"}, "Parameter row coverage")
    for row in list(summary.values()) + list(paired.values()):
        check_role(row)
        require(row["scenario_id"] == "K4_Nsc256_Dfb256", "Mixed scenario")
        for key, value in (("seed", 20261002), ("num_samples", 1600), ("downlink_snr_db", 25)):
            require(number(row, key) == value, f"Mixed formal condition: {key}")
        check_ci(row, "delta_" if "comparison" in row else "sum_rate_")
    for (method, _), row in summary.items():
        for key, value in (("k_users", 4), ("antennas", 32), ("subcarriers", 256),
                           ("feedback_budget", 256)):
            require(number(row, key) == value, f"Mixed summary setting: {key}")
        allocation = (0, 256) if method in ("original", "lite1024") else (64, 0)
        require(tuple(number(row, k) for k in ("allocation_df", "allocation_da")) == allocation,
                f"Mixed allocation: {method}")
        require(number(row, "checkpoint_step") == manifest["methods"][method]["checkpoint_step"],
                f"Summary step differs from manifest: {method}")
    for row in params.values():
        check_role(row)
        require(number(row, "full_parameters") > 0, "Invalid full parameter count")
    for method in ("original", "lite1024"):
        require(0 < number(params[method], "encoder_plus_condition_parameters")
                < number(params[method], "full_parameters"), "Invalid K4 UE count")
    for (comparison, snr), row in paired.items():
        reference = comparison.split("-", 1)[1]
        require(row["method_key"] == "lite1024" and row["reference_method_key"] == reference,
                "Paired direction mismatch")
        lite_mean = number(summary["lite1024", snr], "sum_rate_mean")
        ref_mean = number(summary[reference, snr], "sum_rate_mean")
        # Existing summary reduces float32 tensors; paired CSV reduces per-sample
        # differences in Python double. Do not replace either saved statistic.
        tolerance = 8 * 2**-23 * max(1, abs(lite_mean), abs(ref_mean))
        require(abs((lite_mean - ref_mean) - number(row, "delta_mean")) <= tolerance,
                f"Paired delta inconsistent with rounded float32 summary: {comparison}/{snr}")

    out = []
    for snr in FB_SNRS:
        original, lite = summary["original", snr], summary["lite1024", snr]
        delta = paired["lite1024-original", snr]
        original_mean, lite_mean = number(original, "sum_rate_mean"), number(lite, "sum_rate_mean")
        require(original_mean > 0, "Relative change needs a positive reference rate")
        record = {
            "scenario_id": "K4_Nsc256_Dfb256", "k_users": 4, "antennas": 32,
            "subcarriers": 256, "feedback_budget": 256, "num_subbands": 4,
            "allocation_df": 0, "allocation_da": 256, "feedback_snr_db": snr,
            "downlink_snr_db": 25, "num_samples": 1600, "seed": 20261002,
            "original_sum_rate_mean": original["sum_rate_mean"],
            "lite1024_sum_rate_mean": lite["sum_rate_mean"],
            "lite_minus_original_mean_difference": lite_mean - original_mean,
            "relative_rate_change_percent": 100 * (lite_mean - original_mean) / original_mean,
            **{key: delta[key] for key in ("delta_mean", "delta_se", "delta_ci95_low", "delta_ci95_high")},
            "original_checkpoint_step": original["checkpoint_step"],
            "lite1024_checkpoint_step": lite["checkpoint_step"],
            "original_k4_full_parameters": params["original"]["full_parameters"],
            "lite1024_k4_full_parameters": params["lite1024"]["full_parameters"],
            "original_k4_encoder_plus_condition_parameters": params["original"]["encoder_plus_condition_parameters"],
            "lite1024_k4_encoder_plus_condition_parameters": params["lite1024"]["encoder_plus_condition_parameters"],
            "dataset_role": "independent_held_out_test", "preflight": False,
            "source_run": "paper/data/k4_pure_as_lite1024/" + formal_dir.name,
            **{f"source_{name.split('.')[0]}_sha256_lf": digest for name, digest in source_hashes.items()},
        }
        out.append(record)
    require(before == {name: sha256(path) for name, path in paths.items()}, "Formal files changed during read")
    return out


def tablev_gate(rows):
    """Original instantiated results must match the *published* rounding first."""
    by_allocation = unique_index(rows, lambda r: (r["allocation_df"], r["allocation_da"]))
    require(set(by_allocation) == set(ALLOCATIONS), "Wrong Table-V allocations")
    for allocation, row in by_allocation.items():
        require(f"{row['ue_total_parameters'] / 1e6:.3f}" == TABLEV_PARAMS_M,
                f"Original Table-V parameter mismatch: {row}")
        require(f"{row['macs'] / 1e6:.1f}" == TABLEV_MACS_M[allocation],
                f"Original Table-V MAC mismatch: {row}")


def tablev_config():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from configs.train_config import ExperimentSpec, resolve_experiment
    cfg = resolve_experiment(ExperimentSpec(
        model_name="proposal", train_mode="adaptive", k_users=8,
        subcarriers=128, feedback_budget=128, recipe_name="k8_n128_soft_anchor", seed=42,
    ))
    require((cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS, cfg.D_TOT, cfg.NUM_SUBBANDS,
             cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED) == (8, 32, 128, 128, 4, 16, 128),
            "Not the Table-V configuration")
    require(cfg.GAMMA_GRID == [(16, 0), (12, 32), (8, 64), (4, 96), (0, 128)], "Wrong max-width grid")
    return cfg


def assert_only_ue_ffn_changed(original, lite, torch):
    a, b = original.state_dict(), lite.state_dict()
    require(a.keys() == b.keys(), "Model state keys changed")
    allowed = {f"encoder.{branch}_backbone.layers.{i}.{linear}.{parameter}"
               for branch in ("fdma", "aircomp") for i in range(4)
               for linear in ("linear1", "linear2") for parameter in ("weight", "bias")}
    expected_shapes = {key for key in allowed if not key.endswith("linear2.bias")}
    changed_shapes = {key for key in a if a[key].shape != b[key].shape}
    require(changed_shapes == expected_shapes, f"Unexpected changed shapes: {changed_shapes}")
    require(all(torch.equal(a[key], b[key]) for key in a if key not in allowed),
            "Non-UE-FFN parameters/buffers changed under the same seed")
    for model, width in ((original, 2048), (lite, 1024)):
        require(model.D == 256, "D_MODEL changed")
        for name in ("fdma_backbone", "aircomp_backbone"):
            layers = getattr(model.encoder, name).layers
            require(len(layers) == 4, "UE depth changed")
            require(all((layer.linear1.in_features, layer.linear1.out_features,
                         layer.linear2.in_features, layer.linear2.out_features) == (256, width, width, 256)
                        for layer in layers), "Wrong UE FFN dimensions")
        require([layer.linear1.out_features for layer in model.cross_subband_attn.layers] == [2048],
                "BS cross-subband FFN changed")
        require([layer.ffn[0].out_features for layer in model.layers] == [512] * 5,
                "BS refinement FFN changed")


def profile_tablev():
    """Instantiate both full models on CPU, hook UE only using the old profiler."""
    cfg = tablev_config()
    import torch
    from models.adaptive_hybrid import AdaptiveHybridPrecoder
    from models.adaptive_hybrid_lite import AdaptiveHybridPrecoderLite
    from eval_ue_complexity import proposed_result, unique_parameter_count

    before = {path.name: sha256(path) for path in MODEL_SOURCES}
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    print("Profiling only: temporarily disable MHA fastpath as in eval_ue_complexity.py; restore on exit.")
    rows = []
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.random.fork_rng(devices=[]):
            # CPU generator only: do not touch CUDA RNGs or load any checkpoint.
            torch.random.default_generator.manual_seed(42)
            original = AdaptiveHybridPrecoder(cfg).cpu().eval()

            def measure(model, width):
                result = []
                for df, da in ALLOCATIONS:
                    raw = proposed_result(model, cfg, df, da, 25.0, torch.device("cpu"))
                    result.append({
                        "scenario_id": "K8_Nsc128_Dfb128", "k_users": 8, "antennas": 32,
                        "subcarriers": 128, "feedback_budget": 128, "num_subbands": 4,
                        "model_class": type(model).__name__, "ue_ffn_dim": width,
                        "allocation_df": df, "allocation_da": da,
                        "ue_encoder_parameters": unique_parameter_count([model.encoder]),
                        "ue_condition_parameters": unique_parameter_count([model.cond_embed]),
                        "ue_total_parameters": raw["params"], "macs": raw["macs"],
                    })
                return result

            rows = measure(original, 2048)
            tablev_gate(rows)  # Do not even construct Lite until Original passes.
            torch.random.default_generator.manual_seed(42)
            lite = AdaptiveHybridPrecoderLite(cfg, ue_ffn_dim=1024).cpu().eval()
            assert_only_ue_ffn_changed(original, lite, torch)
            rows += measure(lite, 1024)
            originals = {(row["allocation_df"], row["allocation_da"]): row for row in rows[:3]}
            for row in rows:
                reference = originals[row["allocation_df"], row["allocation_da"]]
                for field, prefix in (("ue_total_parameters", "parameters"), ("macs", "macs")):
                    reduction = reference[field] - row[field]
                    row[f"{prefix}_reduction_absolute"] = reduction
                    row[f"{prefix}_reduction_percent"] = 100 * reduction / reference[field]
                row.update({"counting_scope": "one UE; stored encoder+condition; active-branch learned MACs only",
                            "profiler_source_sha256": sha256(ROOT / "eval_ue_complexity.py"),
                            "original_source_sha256": before[MODEL_SOURCES[0].name],
                            "lite_source_sha256": before[MODEL_SOURCES[1].name],
                            "torch_version": torch.__version__, "device": "cpu",
                            "tablev_original_gate": "passed", "checkpoint_loaded": False})
    finally:
        torch.backends.mha.set_fastpath_enabled(old_fastpath)
        require(before == {path.name: sha256(path) for path in MODEL_SOURCES}, "Model source changed")
    return rows


def save_csv(rows, path):
    require(bool(rows), "Refuse empty output")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    text = stream.getvalue()
    path = Path(path)
    if path.exists():
        require(path.read_text(encoding="utf-8") == text, f"Refuse to overwrite differing output: {path}")
        print(f"Verified unchanged: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
        print(f"Saved: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true", help="CPU structural profile; no checkpoint/channel/BS forward")
    parser.add_argument("--tradeoff", action="store_true", help="Read only the four specified formal artifacts")
    parser.add_argument("--formal-dir", type=Path, default=FORMAL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/data")
    args = parser.parse_args(argv)
    if not args.profile and not args.tradeoff:
        parser.print_help()
        return 0
    try:
        if args.profile:
            rows = profile_tablev()
            save_csv(rows, args.output_dir / "lite1024_complexity_tablev.csv")
            for row in rows:
                print(f"UE FFN{row['ue_ffn_dim']} ({row['allocation_df']},{row['allocation_da']}): "
                      f"{row['ue_total_parameters']:,} parameters; {row['macs']:,} MACs; "
                      f"parameter reduction {row['parameters_reduction_percent']:.4f}%; "
                      f"MAC reduction {row['macs_reduction_percent']:.4f}%")
        if args.tradeoff:
            rows = load_tradeoff(args.formal_dir)
            save_csv(rows, args.output_dir / "lite1024_tradeoff_summary.csv")
            for row in rows:
                print(f"K4 FB={row['feedback_snr_db']}: Original={row['original_sum_rate_mean']}, "
                      f"Lite={row['lite1024_sum_rate_mean']}, paired delta={row['delta_mean']} "
                      f"[{row['delta_ci95_low']}, {row['delta_ci95_high']}], "
                      f"relative mean-rate change={row['relative_rate_change_percent']:.6f}%")
    except (ImportError, ValueError, FileNotFoundError) as exc:
        print(f"STOP: {exc}. No unverified complexity CSV is written; do not install dependencies automatically.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
