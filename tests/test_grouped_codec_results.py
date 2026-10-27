"""CSV-only reporting tests; no graphics backend or model is executed."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tests"):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
import grouped_codec_results as results


def uncertainty(row, mean, sample_se=.1, batch_se=.2):
    for prefix, se in (("delta_", sample_se), ("batch_delta_", batch_se)):
        row.update({prefix + "mean": mean, prefix + "se": se,
                    prefix + "ci95_low": mean - 1.96 * se, prefix + "ci95_high": mean + 1.96 * se})


def fixtures(seed=20261030):
    summary = [dict(method_key=method, gamma_db=g, downlink_snr_db=25,
        training_seed=seed, seed=results.TEST_SEED, num_samples=1600, num_batches=200, batch_size=8,
        preflight=False, noise_variance_N0=.25 * 10 ** (-g / 10),
        sum_rate_mean=20., sum_rate_se=.1, sum_rate_ci95_low=19.804, sum_rate_ci95_high=20.196)
        for method in results.METHODS for g in results.GAMMAS]
    scopes = [("per_gamma", str(g)) for g in results.GAMMAS] + [
        ("low_gamma_mean", "0,5,10"), ("all_gamma_mean", "0,5,10,15,20,25")]
    paired = []
    for name, coefficients in results.CONTRASTS.items():
        for scope, points in scopes:
            row = dict(contrast_id=name, coefficients=json.dumps(coefficients), estimand=scope,
                gamma_points=points, downlink_snr_db=25, training_seed=seed, seed=results.TEST_SEED,
                num_samples=1600, num_batches=200, batch_size=8, training_seeds_pooled=False,
                batch_ci_estimable=True)
            uncertainty(row, .7 if name == "D_minus_E" else -.25)
            paired.append(row)
    return summary, paired


class ReportTests(unittest.TestCase):
    def test_all30_rates56_contrasts_including_negative_interaction(self):
        report = results.interpret(*fixtures(), 20261030)
        self.assertEqual((len(report["all_rates"]), len(report["comparisons"])), (30, 7))
        self.assertTrue(report["engineering_criterion_D_minus_E"])
        for name, selected in report["comparisons"].items():
            self.assertEqual(len(selected["per_gamma"]), 6)
            if name != "D_minus_E": self.assertLess(selected["low_gamma"]["delta_mean"], 0)
        self.assertFalse(report["training_seeds_pooled"])

    def test_engineering_rule_batch_interval_strict_lower_boundary(self):
        summary, paired = fixtures()
        low = next(r for r in paired if r["contrast_id"] == "D_minus_E" and r["estimand"] == "low_gamma_mean")
        high = next(r for r in paired if r["contrast_id"] == "D_minus_E" and r["gamma_points"] == "25")
        uncertainty(low, .5, batch_se=(.5 - .001) / 1.96)
        uncertainty(high, .096, batch_se=.1)
        self.assertTrue(results.interpret(summary, paired, 20261030)["engineering_criterion_D_minus_E"])
        uncertainty(low, .5, batch_se=.5 / 1.96)
        self.assertFalse(results.interpret(summary, paired, 20261030)["engineering_criterion_D_minus_E"])
        uncertainty(low, .5, batch_se=.1)
        uncertainty(high, .0959, batch_se=.1)
        self.assertFalse(results.interpret(summary, paired, 20261030)["engineering_criterion_D_minus_E"])

    def test_missing_duplicate_oldprotocol_or_nonfinite_rejected(self):
        a, b = fixtures()
        cases = [(a[:-1], b), (a + [a[0]], b), (a, b[:-1]), (a, b + [b[0]])]
        for target, key, value in (("summary", "seed", 20261120), ("summary", "preflight", True),
                ("summary", "feedback_snr_db", 0), ("summary", "allocation_da", 256),
                ("summary", "noise_variance_N0", 1), ("summary", "sum_rate_se", -.1),
                ("summary", "sum_rate_mean", float("nan")), ("pairs", "training_seeds_pooled", True),
                ("pairs", "num_samples", 4800), ("pairs", "num_batches", 1),
                ("pairs", "batch_delta_se", float("inf")), ("pairs", "training_seed", 20261031),
                ("pairs", "coefficients", json.dumps({"ordered_all4": 1})),
                ("pairs", "batch_delta_ci95_low", -123)):
            summary, paired = deepcopy((a, b))
            (summary if target == "summary" else paired)[0][key] = value
            cases.append((summary, paired))
        for i, (summary, paired) in enumerate(cases):
            with self.subTest(i=i), self.assertRaises(ValueError):
                results.interpret(summary, paired, 20261030)

    def test_all_contrasts_both_seeds_in_drafts_and_physical_scope_explicit(self):
        interpreted = {str(seed): results.interpret(*fixtures(seed), seed) for seed in results.SEEDS}
        training = {str(seed): {arm: {"best_step": 19500} for arm in results.METHODS} for seed in results.SEEDS}
        english, chinese = results.result_paragraphs(interpreted, training)
        en, zh = "\n".join(english), "\n".join(chinese)
        self.assertIn("-0.2500", en)
        for seed in results.SEEDS: self.assertIn(str(seed), en)
        for name in results.CONTRASTS: self.assertIn(results.CONTRAST_LABELS[name], en)
        for text in ("not the historical received feedback SNR", "not historical Swin",
                     "not training peak memory", "do not establish convergence", "not a common CSI ranking"):
            self.assertIn(text, en)
        self.assertIn("不能拼接旧曲线", zh)
        self.assertIn("不是纯解混消融", zh)

    def test_protocol_contrast_definitions_cannot_drift_from_evaluator(self):
        import evaluate_grouped_codec as entry
        self.assertEqual(results.CONTRASTS, entry.CONTRASTS)
        self.assertEqual(results.TEST_SEED, entry.TEST_SEED)
        self.assertEqual(results.METHODS, entry.METHODS)

    def test_actual_input_hashes_match_across_training_seeds_not_only_seed_label(self):
        rows = [dict(batch_index=i, sample_index_start=8*i, sample_index_stop=8*(i+1),
            H_dl_sha256="a"*64, H_ul_sha256="b"*64, standard_noise_sha256="c"*64) for i in range(200)]
        record = dict(shared_batch_inputs=rows)
        self.assertEqual(results.verify_shared_batch_inputs(record, deepcopy(rows)), rows)
        for defect in ("missing", "duplicate", "noise", "channel", "digest"):
            candidate = deepcopy(record)
            if defect == "missing": candidate["shared_batch_inputs"].pop()
            elif defect == "duplicate": candidate["shared_batch_inputs"][1] = deepcopy(rows[0])
            elif defect == "noise": candidate["shared_batch_inputs"][0]["standard_noise_sha256"] = "d"*64
            elif defect == "channel": candidate["shared_batch_inputs"][0]["H_dl_sha256"] = "d"*64
            else: candidate["shared_batch_inputs"][0]["H_ul_sha256"] = "bad"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                results.verify_shared_batch_inputs(candidate, rows)

    def test_organizer_refuses_failure_before_any_export_or_plotting(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            evaluations = {}
            for seed in results.SEEDS:
                source = folder / str(seed); source.mkdir()
                (source / "manifest.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
                evaluations[str(seed)] = str(source)
            with self.assertRaises(ValueError):
                results.organize_results(folder, evaluations, {str(seed): {} for seed in results.SEEDS})
            self.assertEqual(list(folder.glob("export_*")), [])
            self.assertNotIn("matplotlib.pyplot", sys.modules)


if __name__ == "__main__": unittest.main()
