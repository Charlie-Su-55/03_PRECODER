"""Synthetic evaluation-contract tests: no real checkpoint or GPU evaluation.

The main fixtures are standard-library records for 24 conditions x 8 complete
multi-user channel samples. Tiny CPU tensors only test buffer guards; the
optional actual Lite adapter test uses its own synthetic state, never run data.
"""

import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.baseline_config import make_config
from configs.lite_train_config import lite_config
from configs.train_config import ExperimentSpec, resolve_experiment


FB_SNRS = (0, 5, 10, 15, 20, 25)
SEED = 20261002
SCENARIO_ID = "K4_Nsc256_Dfb256"
SHARED_EVALUATOR_LF_SHA256 = "310a46170d8b3d5d2c93ccc6fb2ff9bf44ee75db31276a88e586c5fff9719327"


def evaluator():
    return importlib.import_module("evaluate_proposal_lite")


def mean_se_ci(values):
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def synthetic_results(plan, count=8):
    """Correlated channel samples make paired vs independent SE distinguishable."""
    summary, samples = [], []
    for entry in plan:
        key = entry["method_key"]
        df, da = entry["allocations"][0]
        for feedback in FB_SNRS:
            rates, nmses, powers = [], [], []
            for index in range(count):
                original = 11.0 + 2.0 * index + feedback / 10.0
                rate = {
                    "original": original,
                    "lite1024": original - 0.6 + 0.2 * index + (feedback - 15) / 10.0,
                    "swin": original + 0.4 + 0.1 * index,
                    "csinet": original - 0.75 + (-1) ** index * 0.2,
                }[key]
                nmse, power = -6.0 + 0.1 * index + 0.02 * feedback, 256.0
                rates.append(rate)
                nmses.append(nmse)
                powers.append(power)
                samples.append({
                    "scenario_id": SCENARIO_ID, "k_users": 4, "subcarriers": 256,
                    "feedback_budget": 256, "model_key": entry["model_key"],
                    "method_key": key, "method_label": entry["label"],
                    "allocation_df": df, "allocation_da": da,
                    "feedback_snr_db": float(feedback), "downlink_snr_db": 25.0,
                    "sample_index": index, "sum_rate": rate, "nmse_db": nmse,
                    "precoder_power": power,
                })
            row = {
                "scenario_id": SCENARIO_ID, "k_users": 4, "antennas": 32,
                "subcarriers": 256, "feedback_budget": 256,
                "model_key": entry["model_key"], "method_key": key,
                "method_label": entry["label"], "checkpoint_kind": "best",
                "checkpoint_step": 49500 if key == "lite1024" else 48000,
                "checkpoint_path": str(entry["path"]), "allocation_df": df,
                "allocation_da": da, "fdma_fraction": 4 * df / 256.0,
                "aircomp_fraction": da / 256.0, "feedback_snr_db": float(feedback),
                "downlink_snr_db": 25.0, "num_samples": count, "seed": SEED,
            }
            for metric, values in (("sum_rate", rates), ("nmse_db", nmses), ("precoder_power", powers)):
                for suffix, value in zip(("mean", "se", "ci95_low", "ci95_high"), mean_se_ci(values)):
                    row[f"{metric}_{suffix}"] = value
            summary.append(row)
    return summary, samples


def training_snapshot(entry, root):
    key = entry["method_key"]
    if key == "lite1024":
        return lite_config(root / "runs_as_screen" / "proposal_lite").to_dict()
    if key == "original":
        cfg = resolve_experiment(ExperimentSpec(
            model_name="proposal", train_mode="specialist", k_users=4,
            subcarriers=256, feedback_budget=256, recipe_name="k4_n256_standard",
            seed=42, allocation=(0, 256),
        ), output_root=root / "runs_as_screen" / "proposal_specialists")
        return cfg.to_dict()
    return make_config(key, "k4_n256_d256", objective="task",
                       output_root=str(root / "runs_as_screen" / "baselines")).to_dict()


def metadata_checkpoints(entry, root):
    """Metadata-only dictionaries, not serialized weights or training evidence."""
    config = training_snapshot(entry, root)
    # Match each unchanged trainer's own fingerprint convention independently.
    fingerprint_config = deepcopy(config)
    if entry["model_key"] != "proposed":
        for name in ("OUTPUT_ROOT", "SAVE_DIR", "BEST_PATH", "SAVE_PATH",
                     "LOG_DIR", "METRICS_PATH", "STATUS_PATH"):
            fingerprint_config.pop(name, None)
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_config, sort_keys=True, ensure_ascii=entry["model_key"] != "proposed",
        default=str,
    ).encode()).hexdigest()[:16]
    best = {"step": 49500 if entry["method_key"] == "lite1024" else 48000,
            "config": deepcopy(config), "config_fingerprint": fingerprint,
            "state": {}, "best_metric": 1.0, "best_score": 1.0}
    if entry["model_key"] == "proposed":
        best["allocation_grid"] = [(0, 256)]
    latest = deepcopy(best)
    latest["step"] = 50000
    side_config = deepcopy(config)
    if entry["model_key"] == "proposed":
        side_config["config_fingerprint"] = fingerprint
    status = {"state": "completed", "step": 50000, "total_steps": 50000}
    directory = entry["path"].parent
    if entry["model_key"] != "proposed":
        directory = entry["path"].parents[2] / "logs" / entry["path"].parent.name
    sidecars = {"config": side_config, "status": status,
                "paths": {"config": str(directory / "config.json"),
                          "status": str(directory / "status.json")}}
    return best, latest, sidecars


class LiteEvaluationPlanAndCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = evaluator().checkpoint_plan(root=self.root)

    def test_four_fixed_best_paths_allocations_and_no_directory_creation(self):
        self.assertEqual([entry["method_key"] for entry in self.plan], ["original", "lite1024", "swin", "csinet"])
        self.assertEqual(len({entry["path"] for entry in self.plan}), 4)
        for entry in self.plan:
            self.assertEqual(entry["path"].name, "best.pth")
            self.assertTrue(entry["path"].is_relative_to(self.root))
            if entry["method_key"] in {"original", "lite1024"}:
                self.assertEqual(entry["model_key"], "proposed")
                self.assertEqual(entry["allocations"], ((0, 256),))
                self.assertIn("k4_n256_standard", str(entry["path"]))
                self.assertIn("df0_da256", str(entry["path"]))
            else:
                self.assertEqual(entry["allocations"], ((64, 0),))
                self.assertIn(f"{entry['method_key']}_task_K4_Nsc256_Dfb256_Df64_sharedUE_seed42",
                              str(entry["path"]))
        self.assertIn("_ueffn1024_", str(self.plan[1]["path"]))
        self.assertNotIn("ueffn1024", str(self.plan[0]["path"]))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_complete_saved_metadata_is_accepted_for_all_four_methods(self):
        for entry in self.plan:
            with self.subTest(method=entry["method_key"]):
                best, latest, sidecars = metadata_checkpoints(entry, self.root)
                evaluator().validate_training_config(entry, best["config"])
                verified = evaluator().verify_checkpoint(entry, best, latest, sidecars)
                self.assertEqual(verified["checkpoint_step"], best["step"])
                self.assertEqual(verified["completion_step"], 50000)
                self.assertEqual(verified["training_status"]["state"], "completed")

    def test_proposal_recipe_allocation_and_maximum_width_mutations_are_rejected(self):
        for entry in self.plan[:2]:
            original = training_snapshot(entry, self.root)
            for section, name, value in (
                ("training_recipe", "val_batch", 64),
                ("training_recipe", "total_steps", 500),
                ("training_recipe", "name", "k4_n256_fb_robust_ft"),
                ("training_recipe", "downlink_snr_db", 25.0),
                ("resolved", "d_f_max_per_user", 0),
                ("resolved", "d_a_max_shared", 64),
                ("experiment", "train_mode", "adaptive"),
            ):
                changed = deepcopy(original)
                changed[section][name] = value
                with self.subTest(method=entry["method_key"], field=(section, name)):
                    with self.assertRaises(ValueError):
                        evaluator().validate_training_config(entry, changed)
            for key, value in (("allocations", [[64, 0]]), ("full_allocation_grid", [[0, 256]])):
                changed = deepcopy(original)
                changed[key] = value
                with self.assertRaises(ValueError):
                    evaluator().validate_training_config(entry, changed)

    def test_lite_architecture_metadata_is_required_and_original_is_not_lite(self):
        lite, original = self.plan[1], self.plan[0]
        config = training_snapshot(lite, self.root)
        for field, value in (("ue_ffn_dim", 2048), ("model_class", "AdaptiveHybridPrecoder"),
                             ("architecture", "original")):
            changed = deepcopy(config)
            changed["architecture_metadata"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluator().validate_training_config(lite, changed)
        changed = deepcopy(config)
        del changed["architecture_metadata"]
        with self.assertRaises(ValueError):
            evaluator().validate_training_config(lite, changed)
        with self.assertRaises(ValueError):
            evaluator().validate_training_config(original, config)

    def test_baseline_task_pure_fdma_and_shared_user_metadata_are_required(self):
        for entry in self.plan[2:]:
            original = training_snapshot(entry, self.root)
            for field, value in (("OBJECTIVE", "reconstruction"), ("D_F", 32),
                                 ("DIM_AIRCOMP_SHARED", 256), ("K_USERS", 8),
                                 ("SHARE_USER_WEIGHTS", False)):
                changed = deepcopy(original)
                changed[field] = value
                with self.subTest(method=entry["method_key"], field=field), self.assertRaises(ValueError):
                    evaluator().validate_training_config(entry, changed)

    def test_best_selection_and_completed_latest_are_distinct_and_checked(self):
        entry = self.plan[1]
        for target, field, value in (("best", "step", 49000), ("best", "step", 50000),
                                     ("latest", "step", 49500), ("status", "step", 49500),
                                     ("status", "state", "paused")):
            best, latest, sidecars = metadata_checkpoints(entry, self.root)
            {"best": best, "latest": latest, "status": sidecars["status"]}[target][field] = value
            with self.subTest(target=target, field=field, value=value), self.assertRaises(ValueError):
                evaluator().verify_checkpoint(entry, best, latest, sidecars)

    def test_checkpoint_missing_or_mismatched_config_grid_and_fingerprint_are_rejected(self):
        entry = self.plan[1]
        for broken_part in ("best_config", "latest_config", "side_config", "fingerprint", "grid", "missing_config"):
            best, latest, sidecars = metadata_checkpoints(entry, self.root)
            if broken_part in {"best_config", "latest_config", "side_config"}:
                config = {"best_config": best["config"], "latest_config": latest["config"],
                          "side_config": sidecars["config"]}[broken_part]
                config["architecture_metadata"]["ue_ffn_dim"] = 2048
            elif broken_part == "fingerprint":
                best["config_fingerprint"] = "wrong-config"
            elif broken_part == "grid":
                best["allocation_grid"] = [(64, 0)]
            else:
                del best["config"]
            with self.subTest(part=broken_part), self.assertRaises(ValueError):
                evaluator().verify_checkpoint(entry, best, latest, sidecars)


class LiteEvaluationResultsTests(unittest.TestCase):
    def setUp(self):
        self.plan = evaluator().checkpoint_plan()
        self.summary, self.samples = synthetic_results(self.plan)

    def test_exact_24_conditions_and_192_channel_samples_validate(self):
        self.assertEqual((len(self.summary), len(self.samples)), (24, 192))
        indexed = evaluator().verify_results(self.summary, self.samples, self.plan, 8)
        self.assertEqual(len(indexed), 24)
        for entry in self.plan:
            for feedback in FB_SNRS:
                self.assertEqual(set(indexed[(entry["method_key"], float(feedback), 25.0)]), set(range(8)))

    def test_missing_duplicate_and_unexpected_summary_conditions_are_rejected(self):
        variants = (self.summary[:-1], self.summary + [deepcopy(self.summary[0])])
        for broken in variants:
            with self.assertRaises(ValueError):
                evaluator().verify_results(broken, self.samples, self.plan, 8)
        changed = deepcopy(self.summary)
        changed[0]["method_key"] = "universal"
        with self.assertRaises(ValueError):
            evaluator().verify_results(changed, self.samples, self.plan, 8)

    def test_missing_duplicate_and_wrong_sample_index_ranges_are_rejected(self):
        for broken in (self.samples[:-1], self.samples + [deepcopy(self.samples[0])]):
            with self.assertRaises(ValueError):
                evaluator().verify_results(self.summary, broken, self.plan, 8)
        changed = deepcopy(self.samples)
        changed[0]["sample_index"] = 8
        with self.assertRaises(ValueError):
            evaluator().verify_results(self.summary, changed, self.plan, 8)

    def test_scenario_allocation_model_snr_seed_and_sample_count_confusion_is_rejected(self):
        changes = (("scenario_id", "K8_Nsc256_Dfb256"), ("k_users", 8), ("antennas", 16),
                   ("subcarriers", 128), ("feedback_budget", 128), ("model_key", "csinet"),
                   ("allocation_df", 64), ("allocation_da", 0), ("feedback_snr_db", 7),
                   ("downlink_snr_db", 0), ("seed", 20260921), ("num_samples", 32))
        for field, value in changes:
            changed = deepcopy(self.summary)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluator().verify_results(changed, self.samples, self.plan, 8)
        for field, value in changes:
            if field not in self.samples[0]:
                continue
            changed = deepcopy(self.samples)
            changed[0][field] = value
            with self.subTest(sample_field=field), self.assertRaises(ValueError):
                evaluator().verify_results(self.summary, changed, self.plan, 8)

    def test_nonfinite_metrics_and_missing_raw_diagnostics_are_rejected(self):
        for field in ("sum_rate", "nmse_db", "precoder_power"):
            for value in (math.nan, math.inf, -math.inf):
                changed = deepcopy(self.samples)
                changed[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    evaluator().verify_results(self.summary, changed, self.plan, 8)
            changed = deepcopy(self.samples)
            del changed[0][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                evaluator().verify_results(self.summary, changed, self.plan, 8)
        for field in ("sum_rate_mean", "sum_rate_se", "sum_rate_ci95_low", "sum_rate_ci95_high",
                      "nmse_db_mean", "precoder_power_mean"):
            changed = deepcopy(self.summary)
            changed[0][field] = math.nan
            with self.subTest(summary_field=field), self.assertRaises(ValueError):
                evaluator().verify_results(changed, self.samples, self.plan, 8)


    def test_missing_required_summary_and_sample_key_fields_raise_value_errors(self):
        required_summary = ("scenario_id", "method_key", "model_key", "k_users", "antennas",
                            "feedback_snr_db", "downlink_snr_db", "allocation_df", "allocation_da",
                            "seed", "num_samples", "sum_rate_se", "nmse_db_ci95_high",
                            "precoder_power_ci95_low")
        required_samples = ("scenario_id", "method_key", "model_key", "k_users", "subcarriers",
                            "feedback_budget", "feedback_snr_db", "downlink_snr_db",
                            "allocation_df", "allocation_da", "sample_index")
        for rows, fields, is_summary in ((self.summary, required_summary, True),
                                        (self.samples, required_samples, False)):
            for field in fields:
                changed = deepcopy(rows)
                del changed[0][field]
                summary, samples = (changed, self.samples) if is_summary else (self.summary, changed)
                with self.subTest(summary=is_summary, missing=field), self.assertRaises(ValueError):
                    evaluator().verify_results(summary, samples, self.plan, 8)


    def test_float32_summary_rounding_is_accepted_without_rewriting_original_values(self):
        rounded = deepcopy(self.summary)
        for row in rounded:
            for metric in ("sum_rate", "nmse_db", "precoder_power"):
                for suffix in ("mean", "se", "ci95_low", "ci95_high"):
                    field = f"{metric}_{suffix}"
                    row[field] = struct.unpack("f", struct.pack("f", row[field]))[0]
        self.assertTrue(any(rounded[index]["sum_rate_mean"] != row["sum_rate_mean"]
                            for index, row in enumerate(self.summary)))
        before_summary, before_samples = deepcopy(rounded), deepcopy(self.samples)
        evaluator().verify_results(rounded, self.samples, self.plan, 8)
        paired = evaluator().paired_rows(rounded, self.samples, self.plan, 8)
        self.assertEqual(rounded, before_summary)
        self.assertEqual(self.samples, before_samples)
        self.assertEqual(paired, evaluator().paired_rows(self.summary, self.samples, self.plan, 8))

    def test_summary_statistics_must_agree_with_the_raw_samples(self):
        for field in ("sum_rate_mean", "sum_rate_se", "sum_rate_ci95_low", "sum_rate_ci95_high"):
            changed = deepcopy(self.summary)
            changed[0][field] += 1.0
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluator().paired_rows(changed, self.samples, self.plan, 8)

    def test_paired_differences_use_same_sample_index_and_unbiased_standard_error(self):
        rows = evaluator().paired_rows(self.summary, list(reversed(self.samples)), self.plan, 8)
        self.assertEqual(len(rows), 12)
        by_key = {(row["comparison"], row["feedback_snr_db"]): row for row in rows}
        self.assertEqual(set(by_key), {(f"lite1024-{reference}", float(fb))
                                      for reference in ("original", "swin") for fb in FB_SNRS})
        indexed = evaluator().verify_results(self.summary, self.samples, self.plan, 8)
        for reference in ("original", "swin"):
            for feedback in FB_SNRS:
                lite = indexed[("lite1024", float(feedback), 25.0)]
                other = indexed[(reference, float(feedback), 25.0)]
                differences = [lite[index]["sum_rate"] - other[index]["sum_rate"] for index in range(8)]
                expected = mean_se_ci(differences)
                row = by_key[(f"lite1024-{reference}", float(feedback))]
                self.assertEqual((row["method_key"], row["reference_method_key"]), ("lite1024", reference))
                self.assertEqual((row["num_samples"], row["seed"], row["scenario_id"]), (8, SEED, SCENARIO_ID))
                self.assertEqual(row["downlink_snr_db"], 25.0)
                for field, value in zip(("delta_mean", "delta_se", "delta_ci95_low", "delta_ci95_high"), expected):
                    self.assertAlmostEqual(row[field], value, places=12)
                independent_se = math.sqrt(mean_se_ci([r["sum_rate"] for r in lite.values()])[1] ** 2
                                           + mean_se_ci([r["sum_rate"] for r in other.values()])[1] ** 2)
                self.assertLess(row["delta_se"], independent_se / 4)

    def test_negative_differences_and_sign_crossings_are_not_filtered_or_clamped(self):
        rows = evaluator().paired_rows(self.summary, self.samples, self.plan, 8)
        original = {row["feedback_snr_db"]: row for row in rows if row["reference_method_key"] == "original"}
        self.assertLess(original[0.0]["delta_mean"], 0.0)
        self.assertGreater(original[25.0]["delta_mean"], 0.0)
        self.assertLess(original[0.0]["delta_ci95_high"], 0.0)
        self.assertEqual(len(rows), 12)

    def test_number_of_independent_samples_is_channels_not_users_or_subcarriers(self):
        for inflated in (8 * 4, 8 * 256):
            with self.subTest(count=inflated), self.assertRaises(ValueError):
                evaluator().verify_results(self.summary, self.samples, self.plan, inflated)
        rows = evaluator().paired_rows(self.summary, self.samples, self.plan, 8)
        self.assertTrue(all(row["num_samples"] == 8 for row in rows))


class LiteEvaluationCliAndPathTests(unittest.TestCase):
    def test_output_is_new_and_contained_under_the_dedicated_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "k4_pure_as_lite1024"
            valid = root / "preflight_20261002T120000Z"
            evaluator().validate_output_path(valid, root=root)
            self.assertFalse(root.exists())
            for invalid in (root, Path(temporary) / "paper_data", root / ".." / "old_evaluation"):
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    evaluator().validate_output_path(invalid, root=root)
            valid.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                evaluator().validate_output_path(valid, root=root)

    def test_mocked_output_symlink_guard_rejects_parent_redirection(self):
        # Inspection-guard unit test only, not evidence that Windows permits
        # creation of actual symlinks; no real filesystem link is created here.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "k4_pure_as_lite1024"
            output = root / "formal_20261002T120000Z"
            original_is_symlink = Path.is_symlink

            def is_symlink(path):
                return path.absolute() == root.absolute() or original_is_symlink(path)

            with mock.patch.object(Path, "is_symlink", autospec=True, side_effect=is_symlink):
                with self.assertRaisesRegex(ValueError, "(?i)symlink|junction"):
                    evaluator().validate_output_path(output, root=root)
            self.assertFalse(root.exists())

    def test_cli_modes_are_exclusive_and_do_not_accept_protocol_overrides(self):
        parser = evaluator().build_arg_parser()
        for arguments in (["--dry-run", "--preflight"], ["--preflight", "--evaluate"],
                          ["--output-dir", "old_results"], ["--seed", "42"],
                          ["--checkpoint", "latest"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parser.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)

    def test_no_arguments_and_both_dry_run_outcomes_are_lightweight_and_read_only(self):
        # Real placeholder bytes are deliberately not a valid torch checkpoint:
        # successful dry-run must inspect only the sidecar JSON, never load weights.
        script = (
            "import contextlib,io,json,sys,hashlib\n"
            "from pathlib import Path\n"
            f"sys.path[:0]=[{str(ROOT)!r},{str(ROOT / 'tests')!r}]\n"
            "import test_lite_evaluation as fixtures\n"
            "import evaluate_proposal_lite as entry\n"
            "from unittest import mock\n"
            "root=Path(sys.argv[1])\n"
            "empty=root/'missing'\n"
            "missing_plan=entry.checkpoint_plan(root=empty)\n"
            "output_existed=entry.OUTPUT_ROOT.exists()\n"
            "stdout,stderr=io.StringIO(),io.StringIO()\n"
            "with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):\n"
            "    assert entry.main([])==0\n"
            "    with mock.patch.object(entry,'checkpoint_plan',return_value=missing_plan):\n"
            "        assert entry.main(['--dry-run'])==1\n"
            "for item in missing_plan:\n"
            "    assert str(item['path']) in stderr.getvalue()\n"
            "    assert str(item['path'].with_name('latest.pth')) in stderr.getvalue()\n"
            "assert not empty.exists()\n"
            "complete=root/'complete'\n"
            "plan=entry.checkpoint_plan(root=complete)\n"
            "for item in plan:\n"
            "    best,latest,sidecars=fixtures.metadata_checkpoints(item,complete)\n"
            "    item['path'].parent.mkdir(parents=True,exist_ok=True)\n"
            "    item['path'].write_bytes(b'not a serialized checkpoint')\n"
            "    item['path'].with_name('latest.pth').write_bytes(b'not serialized either')\n"
            "    for name in ('config','status'):\n"
            "        p=Path(sidecars['paths'][name]); p.parent.mkdir(parents=True,exist_ok=True)\n"
            "        p.write_text(json.dumps(sidecars[name]),encoding='utf-8')\n"
            "before={str(p):p.read_bytes() for p in complete.rglob('*') if p.is_file()}\n"
            "with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):\n"
            "    with mock.patch.object(entry,'checkpoint_plan',return_value=plan):\n"
            "        assert entry.main(['--dry-run'])==0,stderr.getvalue()\n"
            "after={str(p):p.read_bytes() for p in complete.rglob('*') if p.is_file()}\n"
            "assert before==after\n"
            "assert entry.OUTPUT_ROOT.exists()==output_existed\n"
            "blocked={'torch','numpy','tensorflow','sionna','utils','models','baseline_evaluate'}\n"
            "assert not [name for name in sys.modules if name.split('.')[0] in blocked]\n"
            "print(json.dumps({'missing_exit':1,'complete_exit':0,'no_heavy_imports':True,'inputs_unchanged':True}))\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script, temporary],
                                    cwd=temporary, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {"missing_exit": 1, "complete_exit": 0,
                          "no_heavy_imports": True, "inputs_unchanged": True})


class LiteEvaluationSourceTests(unittest.TestCase):
    def functions(self):
        tree = ast.parse((ROOT / "evaluate_proposal_lite.py").read_text(encoding="utf-8"))
        return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def test_shared_evaluator_source_is_unchanged(self):
        source = (ROOT / "baseline_evaluate.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(source).hexdigest(), SHARED_EVALUATOR_LF_SHA256)

    def test_lite_builder_explicitly_uses_1024_strict_loading_eval_and_shared_interface(self):
        function = self.functions()["build_lite_method"]
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        constructors = [node for node in calls if ast.unparse(node.func) == "AdaptiveHybridPrecoderLite"]
        self.assertEqual(len(constructors), 1)
        self.assertEqual({item.arg: ast.literal_eval(item.value) for item in constructors[0].keywords},
                         {"ue_ffn_dim": 1024})
        loads = [node for node in calls if isinstance(node.func, ast.Attribute)
                 and node.func.attr == "load_state_dict"]
        self.assertEqual(len(loads), 1)
        self.assertEqual({item.arg: ast.literal_eval(item.value) for item in loads[0].keywords},
                         {"strict": True})
        call_names = {ast.unparse(node.func) for node in calls}
        self.assertIn("model.eval", call_names)
        self.assertIn("ev.proposal_runtime_config", call_names)
        self.assertIn("ev.validate_checkpoint_scenario", call_names)
        self.assertIn("ev.LoadedMethod", call_names)
        returned = next(node.value for node in function.body if isinstance(node, ast.Return))
        kwargs = {item.arg: item.value for item in returned.keywords}
        self.assertEqual(ast.literal_eval(kwargs["model_key"]), "proposed")
        self.assertEqual(ast.literal_eval(kwargs["method_key"]), "lite1024")
        self.assertEqual(ast.literal_eval(kwargs["allocations"]), ((0, 256),))

    def test_evaluation_delegates_once_without_training_or_backend_changes(self):
        function = self.functions()["run_evaluation"]
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        call_names = [ast.unparse(node.func) for node in calls]
        self.assertEqual(call_names.count("ev.evaluate_scenario"), 1)
        forbidden = {"train_proposal", "generate_training_batch", "optimizer.step",
                     "loss.backward", "torch.set_float32_matmul_precision",
                     "torch.use_deterministic_algorithms", "model.train", "method.model.train"}
        self.assertTrue(forbidden.isdisjoint(call_names))
        self.assertFalse([name for name in call_names if name.endswith(".forward")])
        self.assertFalse([node for node in ast.walk(function) if isinstance(node, ast.Attribute)
                          and isinstance(node.ctx, ast.Store)
                          and ast.unparse(node).startswith(("torch.backends.", "ev."))])
        self.assertGreaterEqual(call_names.count("model_buffer_snapshot"), 2)
        self.assertGreaterEqual(call_names.count("validate_frozen"), 2)
        self.assertIn("ev.build_proposed_method", call_names)
        self.assertIn("ev.build_baseline_method", call_names)


class LiteEvaluationCpuGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise unittest.SkipTest("PyTorch unavailable: tiny CPU guard/adapter checks not run.")

    def test_unchanged_eval_buffers_include_float_complex_integer_and_scalar_values(self):
        torch = self.torch
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2)).eval()
        model.register_buffer("running_pwr", torch.tensor([1.25]))
        model.register_buffer("complex_marker", torch.tensor([1 + 2j], dtype=torch.complex64))
        before = evaluator().model_buffer_snapshot(model)
        evaluator().validate_frozen(before, evaluator().model_buffer_snapshot(model))
        self.assertIn("0.num_batches_tracked", before["buffers"])
        self.assertEqual(before["buffers"]["0.num_batches_tracked"]["shape"], [])
        self.assertIn("complex_marker", before["buffers"])
        self.assertFalse(any(before["training"].values()))

    def test_buffer_value_shape_dtype_or_training_mode_changes_are_rejected(self):
        torch = self.torch
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2)).eval()
        model.register_buffer("running_pwr", torch.tensor([1.0]))
        before = evaluator().model_buffer_snapshot(model)
        with torch.no_grad():
            model.running_pwr.add_(1.0)
        with self.assertRaises(ValueError):
            evaluator().validate_frozen(before, evaluator().model_buffer_snapshot(model))
        for field, value in (("shape", [2]), ("dtype", "torch.float64"), ("sha256", "changed")):
            changed = deepcopy(before)
            changed["buffers"]["running_pwr"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluator().validate_frozen(before, changed)
        model.eval()
        stable = evaluator().model_buffer_snapshot(model)
        model[0].train()
        with self.assertRaises(ValueError):
            evaluator().validate_frozen(stable, evaluator().model_buffer_snapshot(model))

    def test_real_lite_adapter_strict_synthetic_cpu_state_when_dependencies_exist(self):
        # No real checkpoint, forward, channel generation, or evaluation. On the
        # local machine this is explicitly skipped if the normal imports fail.
        try:
            ev = importlib.import_module("baseline_evaluate")
            from models.adaptive_hybrid_lite import AdaptiveHybridPrecoderLite
        except ModuleNotFoundError as exc:
            if exc.name not in {"sionna", "sionna.phy"}:
                raise
            self.skipTest(f"Missing {exc.name!r}: actual Lite adapter synthetic CPU load not run; no bypass.")
        torch = self.torch
        before_rng = torch.get_rng_state().clone()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                entry = evaluator().checkpoint_plan(root=root)[1]
                best, _, _ = metadata_checkpoints(entry, root)
                scenario = ev.Scenario(k_users=4, subcarriers=256, feedback_budget=256)
                cfg = ev.proposal_runtime_config(scenario, best)
                original = AdaptiveHybridPrecoderLite(cfg, ue_ffn_dim=1024).cpu().eval()
                best["state"] = original.state_dict()
                method = evaluator().build_lite_method(ev, scenario, entry["path"], best, torch.device("cpu"))
                self.assertIs(type(method.model), AdaptiveHybridPrecoderLite)
                self.assertEqual(method.allocations, ((0, 256),))
                self.assertEqual(method.config.GAMMA_GRID, [(0, 256)])
                self.assertEqual((method.config.D_F_MAX_PER_USER, method.config.D_A_MAX_SHARED), (64, 256))
                self.assertFalse(any(module.training for module in method.model.modules()))
                for key, value in method.model.state_dict().items():
                    self.assertTrue(torch.equal(value, original.state_dict()[key]), key)
                # Explicitly incompatible synthetic tensors must fail strict load.
                bad = dict(best, state=dict(best["state"]))
                key = "encoder.fdma_backbone.layers.0.linear1.weight"
                bad["state"][key] = bad["state"][key][:-1]
                with self.assertRaises(RuntimeError):
                    evaluator().build_lite_method(ev, scenario, entry["path"], bad, torch.device("cpu"))
        finally:
            torch.set_rng_state(before_rng)


if __name__ == "__main__":
    unittest.main(verbosity=2)
