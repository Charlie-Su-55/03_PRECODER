"""Lightweight orchestration and result interpretation; no model execution."""
import copy
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from cli_contracts import assert_lightweight_cli
from as_receiver_batch_support import save_json
import run_swin_adaptation_batch as batch
import swin_adaptation_results as results


def fixtures(seed=20261030):
    summary = [dict(method_key=method, feedback_snr_db=fb, training_seed=seed,
        seed=results.TEST_SEED, num_samples=1600, preflight=False,
        sum_rate_mean=20., sum_rate_se=.1, sum_rate_ci95_low=19.804,
        sum_rate_ci95_high=20.196) for method in results.METHODS for fb in results.FB]
    scopes = [("per_snr", str(fb)) for fb in results.FB] + [("low_snr_mean", "0,5,10"),
                                                           ("all_snr_mean", "0,5,10,15,20,25")]
    pairs = []
    for method, reference in results.PAIRS:
        for scope, points in scopes:
            delta = -.25 if reference == "swin_adapted" else .7
            pairs.append(dict(method_key=method, reference_method_key=reference,
                estimand=scope, feedback_snr_points=points, training_seed=seed,
                seed=results.TEST_SEED, num_samples=1600, num_batches=200,
                training_seeds_pooled=False, delta_mean=delta, delta_se=.1,
                delta_ci95_low=delta-.196, delta_ci95_high=delta+.196,
                batch_delta_mean=delta, batch_delta_se=.2,
                batch_delta_ci95_low=delta-.392, batch_delta_ci95_high=delta+.392))
    return summary, pairs


class Reporting(unittest.TestCase):
    def test_reporting_rejects_changed_or_wrong_protocol_artifacts(self):
        import evaluate_swin_adaptation as evaluation
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in evaluation.OUTPUT_ARTIFACTS:
                (root / name).write_bytes(b"synthetic artifact, never evaluated")
            manifest = dict(status="completed", freeze_checks={"passed": True},
                protocol=evaluation.protocol(20261030), output_sha256=evaluation.artifact_hashes(root))
            save_json(root / "manifest.json", manifest)
            self.assertEqual(results.verified_result_record(root, 20261030), manifest)
            with self.assertRaises(ValueError):
                results.verified_result_record(root, 20261031)
            (root / evaluation.OUTPUT_ARTIFACTS[0]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                results.verified_result_record(root, 20261030)

    def test_negative_adapted_comparison_and_all_points_retained(self):
        summary, pairs = fixtures()
        report = results.interpret(summary, pairs, 20261030)
        self.assertEqual(len(report["all_rates"]), 36)
        selected = report["comparisons"]["joint_minus_swin_adapted"]
        self.assertEqual(len(selected["per_snr"]), 6)
        self.assertLess(float(selected["low_snr"]["delta_mean"]), 0)
        self.assertFalse(report["training_seeds_pooled"])

    def test_batch_ci_is_not_replaced_with_favorable_sample_ci(self):
        summary, pairs = fixtures()
        for row in pairs:
            if row["reference_method_key"] == "full_control" and row["estimand"] == "low_snr_mean":
                row["batch_delta_ci95_low"] = -.01
        self.assertFalse(results.interpret(summary, pairs, 20261030)["engineering_criterion_vs_full"])

    def test_missing_duplicate_nonfinite_preflight_or_pooling_rejected(self):
        summary, pairs = fixtures()
        cases = [(summary[:-1], pairs), (summary+[summary[0]], pairs),
                 (summary, pairs[:-1]), (summary, pairs+[pairs[0]])]
        for target, key, value in (("summary", "preflight", True), ("summary", "seed", 20261103),
                ("summary", "sum_rate_mean", float("nan")), ("pairs", "training_seeds_pooled", True),
                ("pairs", "batch_delta_se", float("inf"))):
            a, b = copy.deepcopy((summary, pairs))
            (a if target == "summary" else b)[0][key] = value
            cases.append((a, b))
        for index, (a, b) in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                results.interpret(a, b, 20261030)

    def test_draft_reports_step_zero_and_scope_not_historical_budget_matching(self):
        interpretations = {str(seed): results.interpret(*fixtures(seed), seed) for seed in results.SEEDS}
        training = {str(seed): {"best_step": 0} for seed in results.SEEDS}
        en, zh = results.result_paragraphs(interpretations, training)
        self.assertTrue(any("selected step 0" in paragraph for paragraph in en))
        self.assertIn("-0.2500", "\n".join(en))
        self.assertIn("not the historical total budget", "\n".join(en))
        self.assertIn("K4 pure-AS", "\n".join(en))
        self.assertIn("不能替换", "\n".join(zh))


class Controller(unittest.TestCase):
    def test_no_run_summary_does_not_infer_results_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(batch, "OUTPUT_ROOT", root), contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(batch.print_summary(), 0)
            self.assertIn('"status": "not_started"', stream.getvalue())
            self.assertEqual(list(root.iterdir()), [])

    def test_dry_run_is_lightweight_and_trains_only_two_swin_seeds(self):
        assert_lightweight_cli(self, "run_swin_adaptation_batch", [["--dry-run"]],
            "assert len(plans[0]['training'])==2\n"
            "assert {r['seed'] for r in plans[0]['training']}=={20261030,20261031}\n"
            "assert all(r['source_step0_competes_for_best'] for r in plans[0]['training'])\n"
            "assert plans[0]['old_experiments_retrained'] is False\n"
            "assert plans[0]['cleanup'] is False\n"
            "assert {p['seed'] for p in plans[0]['evaluations'].values()}=={20261110}")

    def test_included_500_then_resume_without_proposal_training(self):
        import configs.swin_adapt_train_config as config
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = SimpleNamespace(SAVE_DIR=str(root/"checkpoint"), LOG_DIR=str(root/"logs"),
                SAVE_PATH=str(root/"checkpoint/latest.pth"), BEST_PATH=str(root/"checkpoint/best.pth"),
                CONFIG_PATH=str(root/"logs/config.json"), STATUS_PATH=str(root/"logs/status.json"))
            commands = []
            def child(folder, name, command):
                commands.append(command)
                state, step = ("paused", 500) if "--stop-after" in command else ("completed", 20000)
                Path(cfg.LOG_DIR).mkdir(exist_ok=True)
                save_json(cfg.STATUS_PATH, {"state": state, "step": step}, replace=Path(cfg.STATUS_PATH).exists())
                return str(root/(name+".log"))
            with mock.patch.object(config, "swin_adapt_config", return_value=cfg), \
                    mock.patch.object(batch, "_child", side_effect=child), \
                    mock.patch.object(batch, "_stage_child", return_value={"report": {"completed_updates": 20000}, "_evidence": []}):
                batch._train_one(root, 20261030)
            self.assertEqual(commands[0][-2:], ["--stop-after", "500"])
            self.assertEqual(commands[1][-2:], ["--resume", "--train"])
            self.assertTrue(all(c[3] == "train_swin_adaptation.py" for c in commands))

    def test_completed_swin_is_not_retrained_and_unknown_outputs_preserved(self):
        import configs.swin_adapt_train_config as config
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = SimpleNamespace(SAVE_DIR=str(root/"checkpoint"), LOG_DIR=str(root/"logs"),
                SAVE_PATH=str(root/"checkpoint/latest.pth"), BEST_PATH=str(root/"checkpoint/best.pth"),
                CONFIG_PATH=str(root/"logs/config.json"), STATUS_PATH=str(root/"logs/status.json"))
            Path(cfg.SAVE_DIR).mkdir()
            with mock.patch.object(config, "swin_adapt_config", return_value=cfg), mock.patch.object(batch, "_child") as child:
                with self.assertRaises(RuntimeError):
                    batch._train_one(root, 20261030)
                child.assert_not_called()
            Path(cfg.SAVE_PATH).write_bytes(b"synthetic existence sentinel, never loaded")
            Path(cfg.LOG_DIR).mkdir()
            save_json(cfg.STATUS_PATH, {"state": "completed", "step": 20000})
            with mock.patch.object(config, "swin_adapt_config", return_value=cfg), mock.patch.object(batch, "_child") as child, \
                    mock.patch.object(batch, "_stage_child", return_value={"report": {}, "_evidence": []}):
                batch._train_one(root, 20261030)
                child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
