"""CPU/stdlib checks for complete-MU pairing and lossless ICC result tables."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import twc_v1_k8_icc_aligned_results as results


def sample_rows(samples=8):
    rows = []
    for arm in results.ARMS:
        for gi, gamma in enumerate(results.GAMMAS):
            for i in range(samples):
                delta = (-2. if i < 4 else 1.) + gi * .1
                row = dict(arm=arm, gamma_db=gamma, sample_index=i, batch_index=i // 4,
                    within_batch=i % 4, dl_nmse_linear=1. if i % 2 == 0 else .01,
                    sum_rate_bps_hz=10. + gi + (delta if arm == "hybrid_g4" else 0.),
                    desired_signal_power_mean=.8, interference_power_mean=.1, rzf_power_max_error=0.)
                if arm == "hybrid_g4":
                    row.update(key_energy=0., shared_energy=48., soft_key_nmse_linear=.2)
                rows.append(row)
    return rows


class PairedStatisticsTests(unittest.TestCase):
    def test_linear_nmse_average_precedes_db(self):
        summary = results.summarize_rows(sample_rows(), samples=8)
        row = summary[0]
        self.assertAlmostEqual(row["dl_nmse_linear_mean"], .505)
        self.assertAlmostEqual(row["dl_nmse_db_from_linear_mean"], 10 * math.log10(.505))
        self.assertNotAlmostEqual(row["dl_nmse_db_from_linear_mean"], -10.)

    def test_negative_paired_values_and_batch_ci_are_retained(self):
        paired = results.paired_rows(sample_rows(), samples=8)
        point = next(row for row in paired if row["metric"] == "sum_rate_bps_hz" and
                     row["scope"] == "point" and row["gamma_db"] == 0.)
        self.assertEqual(point["mean"], -.5)
        self.assertEqual(point["n_samples"], 8)
        self.assertEqual(point["n_batches"], 2)
        self.assertAlmostEqual(point["batch_se"], 1.5)
        self.assertLess(point["batch_ci95_low"], 0.)
        self.assertNotEqual(point["batch_se"], point["sample_se"])

    def test_cross_gamma_is_within_mu_not_pseudoreplication(self):
        paired = results.paired_rows(sample_rows(), samples=8)
        all_six = next(row for row in paired if row["metric"] == "sum_rate_bps_hz" and row["scope"] == "all_six")
        low = next(row for row in paired if row["metric"] == "sum_rate_bps_hz" and row["scope"] == "low_0_5_10")
        self.assertAlmostEqual(all_six["mean"], -.25)
        self.assertAlmostEqual(low["mean"], -.4)
        self.assertEqual(all_six["n_samples"], 8)
        self.assertAlmostEqual(all_six["batch_se"], 1.5)

    def test_one_batch_interval_is_not_fabricated_zero(self):
        paired = results.paired_rows(sample_rows(4), samples=4)
        self.assertTrue(all(row["batch_se"] is None and row["batch_ci95_low"] is None and
                            row["batch_ci95_high"] is None for row in paired))

    def test_pairing_rejects_missing_duplicate_nonfinite_and_wrong_batch(self):
        rows = sample_rows()
        for mutation in (lambda r: r.pop(), lambda r: r.append(dict(r[0])),
                         lambda r: r[0].update(sum_rate_bps_hz=float("nan")),
                         lambda r: r[0].update(batch_index=1),
                         lambda r: r[0].update(gamma_db=4)):
            current = [dict(row) for row in rows]
            mutation(current)
            with self.assertRaises(ValueError):
                results.paired_rows(current, samples=8)

    def test_both_input_orders_produce_same_pairing(self):
        rows = sample_rows()
        self.assertEqual(results.paired_rows(rows, samples=8), results.paired_rows(list(reversed(rows)), samples=8))

    def test_individual_training_monitor_does_not_require_other_arm(self):
        rows = [row for row in sample_rows() if row["arm"] == "fdma"]
        with tempfile.TemporaryDirectory() as folder:
            report = results.write_result_tables(folder, rows, samples=8, arms=("fdma",))
            self.assertEqual(len(report["summary"]), 6)
            self.assertEqual(report["paired_deltas"], [])


class ResultFilesTests(unittest.TestCase):
    def test_equal_seed_metadata_does_not_replace_actual_noise_hashes(self):
        from evaluate_twc_v1_k8_icc_aligned import checked_input_hashes
        for value in (None, {"noise_seed": 42}, {"standard_noise": "a"*64},
                      {"standard_noise": "a"*64, "pilot_polarity": "invalid"}):
            with self.assertRaises(ValueError):
                checked_input_hashes(value, H_dl="d"*64, H_ul="e"*64)
        actual = checked_input_hashes(dict(standard_noise="a"*64, pilot_polarity="b"*64),
                                      H_dl="d"*64, H_ul="e"*64)
        self.assertEqual(actual["H_dl"], "d"*64)
        self.assertEqual(actual["standard_noise"], "a"*64)

    def test_gzip_round_trip_preserves_union_blanks_zero_and_metric_values(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            rows = sample_rows()
            results.write_result_tables(out, rows, samples=8)
            reread = results.read_raw_csv(out / "samples.csv.gz")
            self.assertEqual(len(reread), 96)
            fdma = next(row for row in reread if row["arm"] == "fdma")
            hybrid = next(row for row in reread if row["arm"] == "hybrid_g4")
            self.assertIsNone(fdma["key_energy"])
            self.assertEqual(hybrid["key_energy"], 0.)
            self.assertEqual(results.paired_rows(rows, samples=8), results.paired_rows(reread, samples=8))
            before = (out / "summary.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                results.write_result_tables(out, rows, samples=8)
            self.assertEqual((out / "summary.csv").read_bytes(), before)

    def test_package_allows_explicit_small_artifacts_only(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "table.csv").write_text("a\n1\n", encoding="utf-8")
            (output / "checkpoint.pt").write_bytes(b"do not package")
            for path in ("checkpoint.pt", "samples.csv.gz", "../outside.csv"):
                with self.assertRaises(ValueError):
                    results.make_small_package(output, [path])
            record = results.make_small_package(output, ["table.csv"])
            self.assertEqual(record["files"], ["table.csv"])
            with self.assertRaises(FileExistsError):
                results.make_small_package(output, ["table.csv"])

    def test_cli_help_and_dryrun_are_stdlib_and_write_nothing(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "must_not_exist"
            completed = subprocess.run([sys.executable, "-B", "-S",
                str(root / "evaluate_twc_v1_k8_icc_aligned.py"), "--dry-run", "--output-dir", str(target)],
                cwd=root, text=True, capture_output=True, check=True)
            plan = json.loads(completed.stdout)
            self.assertFalse(plan["writes"])
            self.assertFalse(plan["model_loading"])
            self.assertEqual(plan["samples_per_condition"], 1024)
            self.assertEqual(plan["fixed_alpha"], .03)
            self.assertFalse(target.exists())


try:
    import torch
    import numpy as np
except ImportError:
    torch = np = None


@unittest.skipIf(torch is None, "Real Torch/NumPy unavailable; no substitute dependency")
class FrozenComponentTests(unittest.TestCase):
    def test_real_small_module_modes_rng_restored_and_buffer_mutation_rejected(self):
        import random
        from evaluate_twc_v1_k8_icc_aligned import frozen_evaluation
        model = torch.nn.Linear(2, 2)
        ce = torch.nn.Linear(2, 2)
        statistics = torch.nn.Module()
        statistics.register_buffer("prior", torch.ones(2))
        statistics.register_buffer("updates", torch.zeros((), dtype=torch.int64))
        model.train(); ce.eval()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        with frozen_evaluation({"fdma": (model, statistics)}, ce):
            self.assertFalse(model.training)
            self.assertFalse(ce.training)
            random.random(); np.random.random(); torch.rand(2)
        self.assertTrue(model.training)
        self.assertFalse(ce.training)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        with self.assertRaisesRegex(RuntimeError, "statistics"):
            with frozen_evaluation({"fdma": (model, statistics)}, ce):
                statistics.prior.add_(1)
        self.assertTrue(model.training)
        self.assertFalse(ce.training)


if __name__ == "__main__":
    unittest.main()
