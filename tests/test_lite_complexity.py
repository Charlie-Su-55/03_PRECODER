"""Read-only formal-data checks and fail-closed Table-V profiling tests.

These tests do not load checkpoints, generate channels, or evaluate rates.
The optional structural test uses CPU synthetic encoder input only.
"""
from __future__ import annotations

import copy
from importlib.machinery import PathFinder
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_lite1024_complexity", ROOT / "paper/scripts/profile_lite1024_complexity.py"
)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def has_model_dependencies():
    try:
        sionna = importlib.util.find_spec("sionna")
        # Inspect the package search path without importing a legacy Sionna
        # parent (which can otherwise initialize TensorFlow during discovery).
        return bool(importlib.util.find_spec("torch") and sionna
                    and sionna.submodule_search_locations
                    and PathFinder.find_spec("sionna.phy", sionna.submodule_search_locations))
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class FormalTradeoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_rows = {
            name: PROFILE.csv_rows(PROFILE.FORMAL / name)
            for name in PROFILE.FORMAL_FILES if name.endswith(".csv")
        }
        cls.manifest = json.loads((PROFILE.FORMAL / "manifest.json").read_text(encoding="utf-8"))

    def check_mutation_rejected(self, name, change, message):
        rows = copy.deepcopy(self.source_rows)
        change(rows[name])
        with mock.patch.object(PROFILE, "csv_rows", side_effect=lambda path: rows[Path(path).name]):
            with self.assertRaisesRegex(ValueError, message):
                PROFILE.load_tradeoff()

    def test_formal_coverage_and_signed_tradeoff(self):
        before = {name: PROFILE.sha256(PROFILE.FORMAL / name) for name in PROFILE.FORMAL_FILES}
        self.assertEqual(len(self.source_rows[PROFILE.FORMAL_FILES[0]]), 24)
        self.assertEqual(len(self.source_rows[PROFILE.FORMAL_FILES[1]]), 12)
        result = PROFILE.load_tradeoff()
        self.assertEqual([row["feedback_snr_db"] for row in result], list(PROFILE.FB_SNRS))
        saved = {float(row["feedback_snr_db"]): row for row in self.source_rows[PROFILE.FORMAL_FILES[1]]
                 if row["comparison"] == "lite1024-original"}
        for row in result:
            source = saved[row["feedback_snr_db"]]
            for field in ("delta_mean", "delta_se", "delta_ci95_low", "delta_ci95_high"):
                self.assertEqual(row[field], source[field])
            self.assertEqual(row["num_samples"], 1600)
            self.assertFalse(row["preflight"])
            self.assertEqual(row["source_run"], "paper/data/k4_pure_as_lite1024/" + PROFILE.FORMAL.name)
        final = result[-1]
        self.assertLess(float(final["delta_mean"]), 0)
        self.assertLess(float(final["delta_ci95_high"]), 0)
        self.assertAlmostEqual(final["relative_rate_change_percent"], -0.55713, places=4)
        self.assertEqual(before, {name: PROFILE.sha256(PROFILE.FORMAL / name) for name in PROFILE.FORMAL_FILES})

    def test_k8_complexity_config_is_separate_from_k4_formal_performance(self):
        cfg = PROFILE.tablev_config()
        self.assertEqual((cfg.K_USERS, cfg.ANTENNAS, cfg.SUBCARRIERS, cfg.D_TOT, cfg.NUM_SUBBANDS),
                         (8, 32, 128, 128, 4))
        self.assertEqual((cfg.D_F_MAX_PER_USER, cfg.D_A_MAX_SHARED), (16, 128))
        for row in PROFILE.load_tradeoff():
            self.assertEqual((row["k_users"], row["antennas"], row["subcarriers"], row["feedback_budget"]),
                             (4, 32, 256, 256))

    def test_preflight_directory_rejected_before_file_access(self):
        with mock.patch.object(PROFILE, "sha256") as file_hash:
            with self.assertRaisesRegex(ValueError, "specified formal run"):
                PROFILE.load_tradeoff(PROFILE.FORMAL.with_name("preflight_other_run"))
            file_hash.assert_not_called()

    def test_manifest_preflight_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["preflight"] = True
        with mock.patch.object(PROFILE.json, "loads", return_value=manifest):
            with self.assertRaisesRegex(ValueError, "Preflight"):
                PROFILE.load_tradeoff()

    def test_manifest_wrong_seed_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["seed"] = 42
        with mock.patch.object(PROFILE.json, "loads", return_value=manifest):
            with self.assertRaisesRegex(ValueError, "manifest mismatch: seed"):
                PROFILE.load_tradeoff()

    def test_manifest_unexpected_bs_width_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["methods"]["lite1024"]["structure_widths"]["bs_cross_subband"] = [1024]
        with mock.patch.object(PROFILE.json, "loads", return_value=manifest):
            with self.assertRaisesRegex(ValueError, "Unexpected structure widths"):
                PROFILE.load_tradeoff()

    def test_row_preflight_rejected(self):
        self.check_mutation_rejected(PROFILE.FORMAL_FILES[0],
                                     lambda rows: rows[0].update(preflight="True"), "Preflight")

    def test_row_wrong_seed_rejected(self):
        self.check_mutation_rejected(PROFILE.FORMAL_FILES[0],
                                     lambda rows: rows[0].update(seed="42"), "condition: seed")

    def test_nonfinite_values_rejected(self):
        for name, field in ((PROFILE.FORMAL_FILES[0], "sum_rate_mean"),
                            (PROFILE.FORMAL_FILES[1], "delta_se")):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(field=field, value=value):
                    self.check_mutation_rejected(name,
                        lambda rows: rows[0].update({field: value}), "Non-finite")

    def test_duplicate_conditions_rejected(self):
        for name in PROFILE.FORMAL_FILES[:3]:
            with self.subTest(file=name):
                self.check_mutation_rejected(name, lambda rows: rows.append(copy.deepcopy(rows[0])),
                                             "Duplicate key")

    def test_missing_conditions_rejected(self):
        for name in PROFILE.FORMAL_FILES[:2]:
            with self.subTest(file=name):
                self.check_mutation_rejected(name, lambda rows: rows.pop(), "Missing/extra")

    def test_ci_must_match_saved_mean_and_se(self):
        for name, field in ((PROFILE.FORMAL_FILES[0], "sum_rate_ci95_low"),
                            (PROFILE.FORMAL_FILES[1], "delta_ci95_low")):
            with self.subTest(field=field):
                self.check_mutation_rejected(name,
                    lambda rows: rows[0].update({field: str(float(rows[0][field]) - 0.1)}),
                    "1.96-SE interval")

    def test_wrong_paired_direction_rejected(self):
        self.check_mutation_rejected(PROFILE.FORMAL_FILES[1],
                                     lambda rows: rows[0].update(reference_method_key="lite1024"),
                                     "Paired direction mismatch")

    def test_changed_delta_with_valid_ci_rejected(self):
        def change(rows):
            for key in ("delta_mean", "delta_ci95_low", "delta_ci95_high"):
                rows[0][key] = str(float(rows[0][key]) + 0.2)
        self.check_mutation_rejected(PROFILE.FORMAL_FILES[1], change, "Paired delta inconsistent")

    def test_wrong_allocation_rejected(self):
        self.check_mutation_rejected(PROFILE.FORMAL_FILES[0],
                                     lambda rows: rows[0].update(allocation_df="1"), "Mixed allocation")


class TableVGateTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [{"allocation_df": df, "allocation_da": da,
                 "ue_total_parameters": 10756170, "macs": macs}
                for df, da, macs in ((4, 96, 345131776), (16, 0, 174146304), (0, 128, 171051776))]

    def test_original_counts_pass_published_rounding_gate(self):
        PROFILE.tablev_gate(self.rows())

    def test_wrong_original_parameters_rejected(self):
        rows = self.rows()
        rows[0]["ue_total_parameters"] = 10877090  # Different K4 scenario must not substitute.
        with self.assertRaisesRegex(ValueError, "parameter mismatch"):
            PROFILE.tablev_gate(rows)

    def test_wrong_original_macs_rejected(self):
        rows = self.rows()
        rows[0]["macs"] += 1000000
        with self.assertRaisesRegex(ValueError, "MAC mismatch"):
            PROFILE.tablev_gate(rows)

    def test_duplicate_allocation_rejected(self):
        rows = self.rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate key"):
            PROFILE.tablev_gate(rows)

    def test_missing_allocation_rejected(self):
        with self.assertRaisesRegex(ValueError, "Wrong Table-V allocations"):
            PROFILE.tablev_gate(self.rows()[:-1])


class CsvOutputTests(unittest.TestCase):
    def test_provenance_hash_is_portable_across_git_line_endings(self):
        with tempfile.TemporaryDirectory(prefix="lite_complexity_test_") as directory:
            lf, crlf = Path(directory) / "lf.txt", Path(directory) / "crlf.txt"
            lf.write_bytes(b"same\ncontent\n")
            crlf.write_bytes(b"same\r\ncontent\r\n")
            self.assertNotEqual(PROFILE.sha256(lf), PROFILE.sha256(crlf))
            self.assertEqual(PROFILE.text_sha256_lf(lf), PROFILE.text_sha256_lf(crlf))

    def test_identical_csv_is_idempotent_and_different_csv_is_protected(self):
        with tempfile.TemporaryDirectory(prefix="lite_complexity_test_") as directory:
            path = Path(directory) / "output.csv"
            rows = [{"delta": -0.25, "scenario": "synthetic-test"}]
            PROFILE.save_csv(rows, path)
            before, modified = path.read_bytes(), path.stat().st_mtime_ns
            PROFILE.save_csv(rows, path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, modified)
            with self.assertRaisesRegex(ValueError, "Refuse to overwrite"):
                PROFILE.save_csv([{"delta": 0.25, "scenario": "synthetic-test"}], path)
            self.assertEqual(path.read_bytes(), before)

    def test_empty_csv_refused(self):
        with tempfile.TemporaryDirectory(prefix="lite_complexity_test_") as directory:
            path = Path(directory) / "empty.csv"
            with self.assertRaisesRegex(ValueError, "Refuse empty"):
                PROFILE.save_csv([], path)
            self.assertFalse(path.exists())

    def test_no_cli_action_only_shows_help(self):
        with mock.patch.object(PROFILE, "profile_tablev") as profile:
            with mock.patch.object(PROFILE, "load_tradeoff") as tradeoff:
                self.assertEqual(PROFILE.main([]), 0)
                profile.assert_not_called()
                tradeoff.assert_not_called()


@unittest.skipUnless(has_model_dependencies(), "Optional CPU structure profile requires torch and sionna.phy")
class ActualStructureTests(unittest.TestCase):
    def test_original_reproduction_and_lite_ffn_only_change(self):
        before = {path: PROFILE.sha256(path) for path in PROFILE.MODEL_SOURCES}
        rows = PROFILE.profile_tablev()
        self.assertEqual(len(rows), 6)
        PROFILE.tablev_gate(rows[:3])
        self.assertEqual({row["ue_ffn_dim"] for row in rows}, {2048, 1024})
        for row in rows:
            self.assertEqual(row["scenario_id"], "K8_Nsc128_Dfb128")
            self.assertEqual(row["tablev_original_gate"], "passed")
            self.assertFalse(row["checkpoint_loaded"])
        self.assertEqual(before, {path: PROFILE.sha256(path) for path in PROFILE.MODEL_SOURCES})


if __name__ == "__main__":
    unittest.main(verbosity=2)
