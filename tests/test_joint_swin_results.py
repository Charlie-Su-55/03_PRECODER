"""Synthetic CSV-report checks; no plotting, model import, or experiment runs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import joint_swin_results as results


def fixtures(seed=20261030):
    summary = [dict(method_key=method, feedback_snr_db=fb, downlink_snr_db=25,
        training_seed=seed, seed=results.TEST_SEED, num_samples=1600, preflight=False,
        sum_rate_mean=20., sum_rate_se=.1, sum_rate_ci95_low=19.804, sum_rate_ci95_high=20.196)
        for method in results.METHODS for fb in results.FB]
    scopes = [("per_snr", str(fb)) for fb in results.FB] + [
        ("low_snr_mean", "0,5,10"), ("all_snr_mean", "0,5,10,15,20,25")]
    paired = []
    for method, reference in results.PAIRS:
        for scope, points in scopes:
            delta = .7 if reference == "plain_swin256" else -.25
            paired.append(dict(method_key=method, reference_method_key=reference, estimand=scope,
                feedback_snr_points=points, downlink_snr_db=25, training_seed=seed,
                seed=results.TEST_SEED, num_samples=1600, num_batches=200, training_seeds_pooled=False,
                delta_mean=delta, delta_se=.1, delta_ci95_low=delta-.196, delta_ci95_high=delta+.196,
                batch_delta_mean=delta, batch_delta_se=.2,
                batch_delta_ci95_low=delta-.392, batch_delta_ci95_high=delta+.392))
    return summary, paired


class InterpretTests(unittest.TestCase):
    def test_all_methods_pairs_and_negative_outcomes_retained(self):
        report = results.interpret(*fixtures(), 20261030)
        self.assertEqual(len(report["all_rates"]), 36)
        self.assertEqual(len(report["comparisons"]), 5)
        self.assertTrue(report["engineering_criterion_vs_plain"])
        self.assertFalse(report["training_seeds_pooled"])
        for reference in ("spatial_joint", "swin_adapted", "swin", "full_control"):
            selected = report["comparisons"]["joint_swin256_minus_" + reference]
            self.assertLess(selected["low_snr"]["delta_mean"], 0)
            self.assertEqual(len(selected["per_snr"]), 6)

    def test_engineering_rule_uses_batch_interval_and_exact_boundaries(self):
        summary, paired = fixtures()
        low = next(r for r in paired if r["reference_method_key"] == "plain_swin256" and r["estimand"] == "low_snr_mean")
        high = next(r for r in paired if r["reference_method_key"] == "plain_swin256" and r["feedback_snr_points"] == "25")
        low.update(delta_mean=.5, delta_ci95_low=.304, delta_ci95_high=.696,
                   batch_delta_mean=.5, batch_delta_ci95_low=.0001)
        high["batch_delta_ci95_low"] = -.1
        self.assertTrue(results.interpret(summary, paired, 20261030)["engineering_criterion_vs_plain"])
        low["batch_delta_ci95_low"] = 0
        self.assertFalse(results.interpret(summary, paired, 20261030)["engineering_criterion_vs_plain"])
        low["batch_delta_ci95_low"] = .0001
        high["batch_delta_ci95_low"] = -.10001
        self.assertFalse(results.interpret(summary, paired, 20261030)["engineering_criterion_vs_plain"])

    def test_missing_duplicate_nonfinite_preflight_wrong_seed_pooling_rejected(self):
        a, b = fixtures()
        cases = [(a[:-1], b), (a+[a[0]], b), (a, b[:-1]), (a, b+[b[0]])]
        for target, key, value in (("summary", "seed", 20261110), ("summary", "preflight", True),
                ("summary", "sum_rate_se", -.1), ("summary", "sum_rate_mean", float("nan")),
                ("pairs", "training_seeds_pooled", True), ("pairs", "num_batches", 4800),
                ("pairs", "batch_delta_se", float("inf")), ("pairs", "training_seed", 20261031)):
            summary, paired = deepcopy((a, b))
            (summary if target == "summary" else paired)[0][key] = value
            cases.append((summary, paired))
        for i, (summary, paired) in enumerate(cases):
            with self.subTest(i=i), self.assertRaises(ValueError):
                results.interpret(summary, paired, 20261030)

    def test_drafts_both_seeds_all_pairs_and_no_latency_or_sota_claim(self):
        interpreted = {str(seed): results.interpret(*fixtures(seed), seed) for seed in results.SEEDS}
        training = {str(seed): {arm: {"best_step": 19500} for arm in results.METHODS[:2]} for seed in results.SEEDS}
        english, chinese = results.result_paragraphs(interpreted, training)
        self.assertIn("-0.2500", "\n".join(english))
        self.assertIn("20261030", "\n".join(english))
        self.assertIn("20261031", "\n".join(english))
        self.assertIn("not equal-parameter or latency comparisons", "\n".join(english))
        self.assertIn("不能替换", "\n".join(chinese))
        self.assertIn("真值不进入decoder", "\n".join(chinese))

    def test_organizer_rejects_unverified_evidence_before_creating_export(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            evaluations = {}
            for seed in results.SEEDS:
                source = folder / str(seed)
                source.mkdir()
                (source / "manifest.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
                evaluations[str(seed)] = str(source)
            records = {str(seed): {} for seed in results.SEEDS}
            with self.assertRaises(ValueError):
                results.organize_results(folder, evaluations, records)
            self.assertEqual(list(folder.glob("export_*")), [])


if __name__ == "__main__": unittest.main()
