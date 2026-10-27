"""Standard-library protocol/statistics tests; no channels or models are run."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tests"):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
import evaluate_grouped_codec as entry
from cli_contracts import assert_lightweight_cli


def fixtures(count=16, training_seed=20261030):
    samples = []
    for method in entry.METHODS:
        for gamma in entry.GAMMAS:
            for index in range(count):
                x = (index - (count - 1) / 2) / count
                scale = {0: 2., 5: -1., 10: 5., 15: 0., 20: 3., 25: -2.}[gamma]
                offset = dict(legacy_all4=0., ordered_all4=1 + scale * x,
                    legacy_group2=.5 + 2 * x, ordered_group2=2 + 3 * x + .5 * scale * x,
                    ordered_fdma=3.)[method]
                row = dict(entry.sample_identity(method, gamma, index, entry.TEST_SEED, training_seed),
                    sum_rate=20 + index + gamma / 5 + offset, nmse_db=-10., precoder_power=256.,
                    precoder_power_max_error=1e-7, empirical_noise_power=entry.expected_n0(gamma) * (1 + index / count / 10),
                    reliability_mean_active=.5, **{f"energy_ue{k}": 64. for k in range(4)})
                row["receiver_regularization"] = row["noise_variance_N0"] / (64 / row["active_symbols_per_ue"]) + 1e-6
                for prefix in ("recovered", "corrected"):
                    row.update({prefix + "_mse_active": .01, prefix + "_nmse_linear_active": .02,
                                prefix + "_nmse_db_active": 10 * math.log10(.02)})
                samples.append(row)
    summary, _ = entry.summarize_samples(samples, count, entry.TEST_SEED, training_seed)
    return summary, samples


class ProtocolTests(unittest.TestCase):
    def test_five_new_paths_and_explicit_layouts_not_fake_allocations(self):
        plans = entry.checkpoint_plan(20261030)
        self.assertEqual(tuple(p["method_key"] for p in plans), entry.METHODS)
        self.assertEqual(len({p["path"] for p in plans}), 5)
        self.assertEqual([p["active_layout"] for p in plans], ["all4", "all4", "group2", "group2", "fdma"])
        for p in plans:
            self.assertEqual(p["layout_spec"]["occupied_resources"], 256)
            self.assertNotIn("allocation", p)

    def test_new_absolute_noise_and_same_actual_data_protocol_across_training_seeds(self):
        a, b = [entry.protocol(seed) for seed in entry.SEEDS]
        self.assertEqual(a["seed"], 20261210)
        self.assertEqual(a["channel_data_protocol"], b["channel_data_protocol"])
        self.assertEqual(a["noise_address"], b["noise_address"])
        self.assertEqual(a["noise_address"]["omits"], ["arm", "training_seed", "gamma"])
        self.assertEqual(a["energy_per_frame_per_ue"], 64)
        self.assertNotIn("feedback_snr_db", a)
        for gamma in entry.GAMMAS:
            self.assertEqual(entry.expected_n0(gamma), .25 * 10 ** (-gamma / 10))
        self.assertEqual(entry.protocol(20261030, True)["num_samples"], 8)
        self.assertEqual(entry.protocol(20261030, True)["seed"], 20261211)

    def test_dryrun_help_no_heavy_import_weight_read_or_output(self):
        assert_lightweight_cli(self, "evaluate_grouped_codec",
            [["--seed", str(seed), "--dry-run"] for seed in entry.SEEDS],
            "assert all(p['models_not_imported'] and p['checkpoint_bytes_not_read'] for p in plans)\n"
            "assert all(not p['old_protocol_curves_allowed'] for p in plans)")

    def test_no_historical_initialization_and_strict_load_syntax(self):
        tree = ast.parse((ROOT / "evaluate_grouped_codec.py").read_text(encoding="utf-8"))
        loaders = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict"]
        self.assertEqual(len(loaders), 1)
        self.assertTrue(any(k.arg == "strict" and isinstance(k.value, ast.Constant) and k.value.value is True for k in loaders[0].keywords))
        self.assertNotIn("migrate", (ROOT / "evaluate_grouped_codec.py").read_text(encoding="utf-8"))


class CheckpointMetadataTests(unittest.TestCase):
    def fixture(self, arm="ordered_group2", selected_step=19500):
        from test_train_grouped_codec import resume_metadata
        from train_grouped_codec import fingerprint
        cfg = entry.grouped_codec_config(arm, 20261030)
        latest = resume_metadata(cfg, 20000)
        best = {name: deepcopy(latest[name]) for name in
                ("state", "config", "config_fingerprint", "layout_spec", "execution_protocol", "step", "best_metric", "val_table")}
        best["step"] = selected_step
        sidecars = dict(config=dict(cfg.to_dict(), config_fingerprint=fingerprint(cfg)),
                        status=dict(state="completed", step=20000, experiment=cfg.EXP_NAME,
                                    best_metric_name="mean_all", best_metric=best["best_metric"]))
        return cfg, best, latest, sidecars

    def test_all_five_random20k_arms_best_can_precede_latest(self):
        for arm in entry.METHODS:
            cfg, best, latest, sidecars = self.fixture(arm)
            record = entry.verify_training_metadata(arm, cfg.SEED, best, latest, sidecars)
            self.assertEqual((record["selected_best_step"], record["latest_step"]), (19500, 20000))
            self.assertTrue(record["random_initialization"])
            self.assertFalse(record["historical_weights_loaded"])

    def test_zero_best_incomplete_old_layout_cross_arm_selection_sidecars_rejected(self):
        for defect in ("best0", "incomplete", "old_allocation", "layout", "arm", "fingerprint",
                       "protocol", "best_mean", "status", "sidecar", "scheduler"):
            cfg, best, latest, sidecars = self.fixture()
            if defect == "best0": best["step"] = 0
            elif defect == "incomplete": latest["step"] = 19500
            elif defect == "old_allocation": latest["allocation_grid"] = [(0, 256)]
            elif defect == "layout": best["layout_spec"]["layout"] = "fdma"
            elif defect == "arm": latest["config"]["arm"] = "legacy_all4"
            elif defect == "fingerprint": latest.pop("config_fingerprint")
            elif defect == "protocol": best["execution_protocol"]["version"] = "old_received_SNR"
            elif defect == "best_mean": best["val_table"]["layout0_gamma0"] += 1
            elif defect == "status": sidecars["status"]["state"] = "paused"
            elif defect == "sidecar": sidecars["config"]["initialization"]["source"] = "old_checkpoint"
            else: latest["scheduler"]["total_steps"] = 20500
            with self.subTest(defect=defect), self.assertRaises((ValueError, RuntimeError)):
                entry.verify_training_metadata(cfg.arm, cfg.SEED, best, latest, sidecars)


class StatisticsTests(unittest.TestCase):
    def compute(self, count=16, training_seed=20261030):
        return entry.paired_rows(*fixtures(count, training_seed), count, entry.TEST_SEED, training_seed)

    def test_all30_conditions56_contrasts_and_signed_negative_primary(self):
        summary, _ = fixtures()
        paired, raw = self.compute()
        self.assertEqual((len(summary), len(paired), len(raw)), (30, 56, 56 * 16))
        low = next(r for r in paired if r["contrast_id"] == "D_minus_E" and r["estimand"] == "low_gamma_mean")
        self.assertAlmostEqual(low["delta_mean"], -1)
        self.assertEqual(low["role"], "primary")

    def test_fourway_interaction_formed_before_uncertainty(self):
        paired, raw = self.compute()
        gamma = next(r for r in paired if r["contrast_id"].startswith("interaction") and r["gamma_db"] == 0)
        # Shared-sample covariance cancels exactly at gamma0; summing component SEs is wrong.
        self.assertAlmostEqual(gamma["delta_mean"], .5)
        self.assertLess(gamma["delta_se"], 1e-14)
        selected = [r["delta"] for r in raw if r["contrast_id"].startswith("interaction") and r["gamma_points"] == "0"]
        self.assertTrue(all(abs(value - .5) < 1e-13 for value in selected))

    def test_low_gamma_average_within_channel_then_batch_not4800_independent_rows(self):
        paired, _ = self.compute()
        low = next(r for r in paired if r["contrast_id"] == "D_minus_E" and r["estimand"] == "low_gamma_mean")
        values = [-1 + 4 * (i - 7.5) / 16 for i in range(16)]
        for name, value in entry.sample_statistics(values).items():
            self.assertAlmostEqual(low["delta_" + name], value)
        batches = [statistics.fmean(values[:8]), statistics.fmean(values[8:])]
        for name, value in entry.sample_statistics(batches).items():
            self.assertAlmostEqual(low["batch_delta_" + name], value)
        self.assertEqual((low["num_samples"], low["num_batches"]), (16, 2))

    def test_preflight_batch_ci_not_estimable_and_training_seeds_not_pooled(self):
        for seed in entry.SEEDS:
            paired, _ = self.compute(8, seed)
            self.assertTrue(all(r["batch_delta_se"] is None and r["batch_delta_ci95_low"] is None for r in paired))
            self.assertTrue(all(not r["batch_ci_estimable"] and r["training_seed"] == seed and not r["training_seeds_pooled"] for r in paired))

    def test_shuffled_rows_pair_by_complete_key(self):
        summary, samples = fixtures()
        expected, raw = self.compute()
        actual, reordered = entry.paired_rows(list(reversed(summary)), list(reversed(samples)), 16, entry.TEST_SEED, 20261030)
        self.assertEqual(actual, expected)
        self.assertEqual(reordered, raw)

    def test_missing_duplicate_nonfinite_oldprotocol_energy_noise_layout_rejected(self):
        for defect in ("missing", "duplicate", "nan", "old", "allocation", "energy", "noise", "negative_noise", "layout", "seed", "N0", "ridge", "bandpower", "power"):
            summary, samples = fixtures()
            if defect == "missing": samples.pop()
            elif defect == "duplicate": samples.append(deepcopy(samples[0]))
            elif defect == "nan": samples[0]["sum_rate"] = math.nan
            elif defect == "old": samples[0]["feedback_snr_db"] = 0
            elif defect == "allocation": samples[0]["allocation_df"] = 0
            elif defect == "energy": samples[0]["energy_ue2"] = 63.
            elif defect == "noise": samples[0]["empirical_noise_power"] *= 2
            elif defect == "negative_noise": samples[0]["empirical_noise_power"] = -1
            elif defect == "layout": samples[0]["active_layout"] = "fdma"
            elif defect == "seed": samples[0]["training_seed"] = 20261031
            elif defect == "N0": samples[0]["noise_variance_N0"] *= 2
            elif defect == "ridge": samples[0]["receiver_regularization"] *= 2
            elif defect == "bandpower": samples[0]["precoder_power"] = 1
            else: samples[0]["precoder_power_max_error"] = .01
            with self.subTest(defect=defect), self.assertRaises((ValueError, RuntimeError)):
                entry.paired_rows(summary, samples, 16, entry.TEST_SEED, 20261030)

    def test_summary_cannot_disagree_with_raw_or_add_oracle(self):
        for defect in ("number", "oracle", "duplicate", "missing"):
            summary, samples = fixtures()
            if defect == "number": summary[0]["sum_rate_mean"] += 1
            elif defect == "oracle": summary[0]["method_key"] = "oracle"
            elif defect == "duplicate": summary.append(deepcopy(summary[0]))
            else: summary.pop()
            with self.subTest(defect=defect), self.assertRaises((ValueError, RuntimeError)):
                entry.paired_rows(summary, samples, 16, entry.TEST_SEED, 20261030)


class OutputTests(unittest.TestCase):
    def test_existing_output_refused_and_artifact_hashes_detect_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = entry.shared.create_output(20261030, output_dir=root / "new", root=root)
            with self.assertRaises((ValueError, FileExistsError)):
                entry.shared.create_output(20261030, output_dir=output, root=root)
            for name in entry.OUTPUT_ARTIFACTS: (output / name).write_bytes(b"synthetic fixture\n")
            original = entry.artifact_hashes(output)
            self.assertEqual(set(original), set(entry.OUTPUT_ARTIFACTS))
            (output / entry.OUTPUT_ARTIFACTS[0]).write_bytes(b"changed synthetic fixture\n")
            self.assertNotEqual(original, entry.artifact_hashes(output))
            (output / entry.OUTPUT_ARTIFACTS[1]).unlink()
            with self.assertRaises(ValueError): entry.artifact_hashes(output)


if __name__ == "__main__": unittest.main()
