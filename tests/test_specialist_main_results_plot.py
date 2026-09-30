"""CSV contract and lightweight rendering checks; no model dependencies."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper/scripts/plot_specialist_main_results.py"
SPEC = importlib.util.spec_from_file_location("specialist_main_results_plot", SCRIPT)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


class SpecialistPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envelope = plot.load_summary(plot.DEFAULT_ENVELOPE, "envelope")
        cls.sweep = plot.load_summary(plot.DEFAULT_SWEEP, "sweep")

    def mutate_csv(self, kind, mutation):
        source = plot.DEFAULT_ENVELOPE if kind == "envelope" else plot.DEFAULT_SWEEP
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        mutation(rows)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        destination = Path(folder.name) / "summary.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return destination

    def test_every_source_rate_and_ci_is_preserved(self):
        for source, loaded in ((plot.DEFAULT_ENVELOPE, self.envelope), (plot.DEFAULT_SWEEP, self.sweep)):
            with source.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    key = (row["method_key"], (int(row["allocation_df"]), int(row["allocation_da"])),
                           float(row["feedback_snr_db"]))
                    for field in plot.RATE_FIELDS:
                        self.assertEqual(loaded[key][field], float(row[field]))
        self.assertEqual(len(self.envelope), 12)
        self.assertEqual(len(self.sweep), 30)

    def test_missing_grid_point_rejected(self):
        for kind in ("envelope", "sweep"):
            with self.subTest(kind=kind):
                path = self.mutate_csv(kind, lambda rows: rows.pop())
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    plot.load_summary(path, kind)

    def test_duplicate_rejected(self):
        for kind in ("envelope", "sweep"):
            with self.subTest(kind=kind):
                path = self.mutate_csv(kind, lambda rows: rows.append(dict(rows[0])))
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    plot.load_summary(path, kind)

    def test_ci_mismatch_rejected(self):
        path = self.mutate_csv("envelope", lambda rows: rows[0].update(
            sum_rate_ci95_high=str(float(rows[0]["sum_rate_ci95_high"]) + 0.1)))
        with self.assertRaisesRegex(ValueError, "CI mismatch"):
            plot.load_summary(path, "envelope")

    def test_nonfinite_rate_rejected(self):
        for bad in ("nan", "inf", "-inf"):
            with self.subTest(value=bad):
                path = self.mutate_csv("sweep", lambda rows: rows[0].update(sum_rate_mean=bad))
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    plot.load_summary(path, "sweep")

    def test_protocol_and_allocation_rejected(self):
        for field, bad in (("k_users", "8"), ("antennas", "64"), ("subcarriers", "128"),
                           ("feedback_budget", "128"), ("num_samples", "256"),
                           ("seed", "20360928"), ("downlink_snr_db", "20"),
                           ("feedback_snr_db", "24"), ("allocation_da", "256"),
                           ("checkpoint_step", "20000")):
            with self.subTest(field=field):
                path = self.mutate_csv("envelope", lambda rows: rows[0].update({field: bad}))
                with self.assertRaises(ValueError):
                    plot.load_summary(path, "envelope")

    def test_deterministic_plot_order_and_separate_sources(self):
        benchmark = plot.benchmark_rows(self.envelope)
        self.assertEqual([round(r["sum_rate_mean"], 3) for r in benchmark],
                         [21.517, 28.545, 28.948, 29.296])
        pure_as = plot.sweep_rows(self.sweep, "specialist_df0_da256")
        self.assertEqual([round(r["sum_rate_mean"], 3) for r in pure_as],
                         [11.282, 16.058, 21.746, 26.117, 28.322, 29.257])
        self.assertNotEqual(benchmark[-1]["sum_rate_mean"], pure_as[-1]["sum_rate_mean"])
        universal = plot.flexibility_rows(self.envelope, universal=True)
        self.assertEqual([round(r["sum_rate_mean"], 3) for r in universal],
                         [20.690, 24.901, 26.262, 27.240, 27.398])
        reversed_input = dict(reversed(list(self.envelope.items())))
        self.assertEqual(plot.benchmark_rows(reversed_input), benchmark)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is optional for CSV validation")
    def test_render_smoke_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            files = plot.render(self.envelope, self.sweep, Path(folder))
            self.assertEqual(len(files), 6)
            for path in files:
                self.assertGreater(path.stat().st_size, 1000)
            before = {p: p.read_bytes() for p in files}
            with self.assertRaises(FileExistsError):
                plot.render(self.envelope, self.sweep, Path(folder))
            self.assertEqual(before, {p: p.read_bytes() for p in files})


if __name__ == "__main__":
    unittest.main()
