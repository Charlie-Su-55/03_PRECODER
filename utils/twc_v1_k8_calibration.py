"""Calibration-grid access for the frozen K8 UL CE estimator.

The grid contains empirical complex channel-error variance from the independent
``ce_calibration`` role. Bilinear interpolation is in feedback gamma (dB) and
per-UE pilot energy; this is a receiver approximation, not a posterior claim.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import torch


class TWCv1K8CECalibration:
    def __init__(self, rows, *, source="in_memory"):
        parsed = {}
        for row in rows:
            gamma = float(row["gamma_db"])
            ep = float(row["pilot_energy_per_ue"])
            value = float(row["error_variance_complex"])
            if not (math.isfinite(gamma) and math.isfinite(ep)
                    and math.isfinite(value) and value >= 0):
                raise ValueError("Calibration entries must be finite with nonnegative error variance")
            key = (gamma, ep)
            if key in parsed:
                raise ValueError(f"Duplicate calibration condition {key}")
            parsed[key] = value
        gammas = tuple(sorted({key[0] for key in parsed}))
        energies = tuple(sorted({key[1] for key in parsed}))
        expected = {(g, e) for g in gammas for e in energies}
        if not gammas or not energies or set(parsed) != expected:
            raise ValueError("Calibration grid must be complete and rectangular")
        if len(gammas) < 2 or len(energies) < 2:
            raise ValueError("Calibration interpolation requires at least two gamma and Ep nodes")
        self.gammas, self.energies, self.values = gammas, energies, parsed
        self.source = str(source)

    @classmethod
    def from_csv(cls, path):
        path = Path(path).expanduser().resolve()
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        result = cls(rows, source=str(path))
        if len(result.values) != 18:
            raise ValueError(f"Expected the frozen 18-condition K8 calibration grid, got {len(result.values)}")
        return result

    @staticmethod
    def _bracket(nodes, value):
        if not math.isfinite(float(value)) or value < nodes[0] or value > nodes[-1]:
            raise ValueError(f"Condition {value} lies outside calibration range [{nodes[0]}, {nodes[-1]}]")
        for i in range(len(nodes) - 1):
            if nodes[i] <= value <= nodes[i + 1]:
                weight = (value - nodes[i]) / (nodes[i + 1] - nodes[i])
                return nodes[i], nodes[i + 1], weight
        return nodes[-1], nodes[-1], 0.0

    def error_variance(self, gamma_db, pilot_energy_per_ue, *, device=None,
                       dtype=torch.float32):
        g0, g1, wg = self._bracket(self.gammas, float(gamma_db))
        e0, e1, we = self._bracket(self.energies, float(pilot_energy_per_ue))
        v00, v01 = self.values[g0, e0], self.values[g0, e1]
        v10, v11 = self.values[g1, e0], self.values[g1, e1]
        low = v00 * (1.0 - we) + v01 * we
        high = v10 * (1.0 - we) + v11 * we
        value = low * (1.0 - wg) + high * wg
        return torch.tensor(value, dtype=dtype, device=device)

    def metadata(self):
        return {"source": self.source, "role": "ce_calibration",
            "gamma_nodes_db": list(self.gammas), "pilot_energy_nodes_per_ue": list(self.energies),
            "conditions": len(self.values), "statistic": "mean |H_UL_hat-H_UL|^2 per complex coefficient",
            "interpolation": "bilinear in gamma dB and pilot energy; scalar isotropic channel-error approximation"}
