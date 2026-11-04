"""Configuration surface for the TWC 1.0 K8 protocol baseline.

Energy candidates, model recipe, channel-data seeds and train/validation/test
namespaces are intentionally left to subsequent plan stages.
"""
from dataclasses import dataclass
import math
import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_NAME = "_twc_v1_k8_protocol"
_MODULE_PATH = REPO_ROOT / "utils" / "twc_v1_k8_protocol.py"
if _MODULE_NAME in sys.modules:
    _protocol = sys.modules[_MODULE_NAME]
    if Path(_protocol.__file__).resolve() != _MODULE_PATH.resolve():
        raise RuntimeError("TWC K8 protocol module identity collision")
else:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if _spec is None or _spec.loader is None:
        raise RuntimeError("Cannot load TWC K8 protocol module")
    _protocol = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _protocol
    try:
        _spec.loader.exec_module(_protocol)
    except BaseException:
        del sys.modules[_MODULE_NAME]
        raise

(
    DOWNLINK_SNR_DB,
    EVALUATION_FEEDBACK_GAMMA_DB,
    K_USERS,
    KEY_DIM_PER_USER,
    N_ANTENNAS,
    N_SUBCARRIERS,
    PILOTS_PER_USER,
    SUPPORTED_GROUP_COUNTS,
    TOTAL_FRAME_ENERGY_PER_USER,
    TOTAL_PILOT_RE,
    TOTAL_RE,
    TRAIN_FEEDBACK_GAMMA_DB,
    FDMALayout,
    HybridLayout,
    default_hybrid_layout,
    enumerate_partitions,
    feedback_noise_variance,
    protocol_metadata,
) = tuple(getattr(_protocol, name) for name in (
    "DOWNLINK_SNR_DB", "EVALUATION_FEEDBACK_GAMMA_DB", "K_USERS",
    "KEY_DIM_PER_USER", "N_ANTENNAS", "N_SUBCARRIERS", "PILOTS_PER_USER",
    "SUPPORTED_GROUP_COUNTS", "TOTAL_FRAME_ENERGY_PER_USER", "TOTAL_PILOT_RE",
    "TOTAL_RE", "TRAIN_FEEDBACK_GAMMA_DB", "FDMALayout", "HybridLayout",
    "default_hybrid_layout", "enumerate_partitions", "feedback_noise_variance",
    "protocol_metadata",
))


@dataclass(frozen=True)
class EnergySplit:
    """Per-UE Ep/Ekey/EAS point; values must be supplied by a later plan step."""
    pilot: float
    key: float
    shared: float

    def __post_init__(self):
        values = (self.pilot, self.key, self.shared)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("Ep, Ekey and EAS must be real scalars")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Ep, Ekey and EAS must be finite")
        if any(value <= 0 for value in values):
            raise ValueError("Hybrid energy split must retain positive pilot, key and shared energy")
        if abs(sum(values) - TOTAL_FRAME_ENERGY_PER_USER) > 1e-9:
            raise ValueError("Ep + Ekey + EAS must equal 64 per UE")


@dataclass(frozen=True)
class TWCv1K8Config:
    """Fixed K8 geometry/metrics; intentionally not a training-ready config."""
    group_count: int
    groups: tuple | None = None

    def __post_init__(self):
        if type(self.group_count) is not int or self.group_count not in SUPPORTED_GROUP_COUNTS:
            raise ValueError(f"group_count must be one of {SUPPORTED_GROUP_COUNTS}")
        if self.groups is None:
            object.__setattr__(self, "groups", default_hybrid_layout(self.group_count).groups)
        layout = HybridLayout(self.groups)
        if layout.group_count != self.group_count:
            raise ValueError("group partition count does not match group_count")

    @property
    def layout(self):
        return HybridLayout(self.groups)

    @property
    def metadata(self):
        result = protocol_metadata()
        result["active_layout"] = self.layout.to_dict()
        result["fdma_control_layout"] = FDMALayout().to_dict()
        return result


def _partitions_of_pairs(users):
    """Generate all pairings of a tuple of users exactly once."""
    if not users:
        yield ()
        return
    first = users[0]
    for partner in users[1:]:
        rest = tuple(user for user in users[1:] if user != partner)
        for suffix in _partitions_of_pairs(rest):
            yield ((first, partner),) + suffix


def all_groupings(group_count):
    """Return the complete unlabeled grouping set required for CPU scoring."""
    if group_count == 1:
        return enumerate_partitions(1)
    if group_count == 2:
        return enumerate_partitions(2)
    if group_count == 4:
        return tuple(_partitions_of_pairs(tuple(range(K_USERS))))
    raise ValueError(f"group_count must be one of {SUPPORTED_GROUP_COUNTS}")


__all__ = [
    "DOWNLINK_SNR_DB", "EVALUATION_FEEDBACK_GAMMA_DB", "EnergySplit",
    "FDMALayout", "HybridLayout", "K_USERS", "KEY_DIM_PER_USER",
    "N_ANTENNAS", "N_SUBCARRIERS", "PILOTS_PER_USER", "SUPPORTED_GROUP_COUNTS",
    "TOTAL_FRAME_ENERGY_PER_USER", "TOTAL_PILOT_RE", "TOTAL_RE",
    "TRAIN_FEEDBACK_GAMMA_DB", "TWCv1K8Config", "all_groupings",
    "default_hybrid_layout", "enumerate_partitions", "feedback_noise_variance",
    "protocol_metadata",
]
