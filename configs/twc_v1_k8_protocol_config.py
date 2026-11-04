"""Configuration surface for the TWC 1.0 K8 protocol baseline.

The finite energy grid is a screening candidate set, not a chosen policy or an
optimal allocation. Model recipe and train/validation/test namespaces remain
for subsequent plan stages.
"""
from dataclasses import dataclass
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
    EnergySplit,
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
    "DOWNLINK_SNR_DB", "EnergySplit", "EVALUATION_FEEDBACK_GAMMA_DB", "K_USERS",
    "KEY_DIM_PER_USER", "N_ANTENNAS", "N_SUBCARRIERS", "PILOTS_PER_USER",
    "SUPPORTED_GROUP_COUNTS", "TOTAL_FRAME_ENERGY_PER_USER", "TOTAL_PILOT_RE",
    "TOTAL_RE", "TRAIN_FEEDBACK_GAMMA_DB", "FDMALayout", "HybridLayout",
    "default_hybrid_layout", "enumerate_partitions", "feedback_noise_variance",
    "protocol_metadata",
))


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
        result["energy_screen_grid"] = [split.to_dict() for split in candidate_energy_splits()]
        result["pilot_energy_policy"] = "select Ep from {8,16,24} by feedback-gamma validation lookup; lookup not selected yet"
        result["key_shared_policy"] = "screen K:EAS ratios 1:3 and 1:1 on validation data"
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
    if type(group_count) is not int:
        raise ValueError(f"group_count must be one of {SUPPORTED_GROUP_COUNTS}")
    if group_count == 1:
        return enumerate_partitions(1)
    if group_count == 2:
        return enumerate_partitions(2)
    if group_count == 4:
        return tuple(_partitions_of_pairs(tuple(range(K_USERS))))
    raise ValueError(f"group_count must be one of {SUPPORTED_GROUP_COUNTS}")


PILOT_ENERGY_CANDIDATES = (8.0, 16.0, 24.0)
KEY_SHARED_RATIO_CANDIDATES = ((1, 3), (1, 1))


def candidate_energy_splits():
    """Small validation-screen grid; it makes no claim of an optimal choice."""
    splits = []
    for pilot_energy in PILOT_ENERGY_CANDIDATES:
        data_energy = TOTAL_FRAME_ENERGY_PER_USER - pilot_energy
        for key_weight, shared_weight in KEY_SHARED_RATIO_CANDIDATES:
            denominator = key_weight + shared_weight
            splits.append(EnergySplit(
                pilot_energy,
                data_energy * key_weight / denominator,
                data_energy * shared_weight / denominator,
            ))
    return tuple(splits)


__all__ = [
    "DOWNLINK_SNR_DB", "EVALUATION_FEEDBACK_GAMMA_DB", "EnergySplit",
    "FDMALayout", "HybridLayout", "K_USERS", "KEY_DIM_PER_USER",
    "N_ANTENNAS", "N_SUBCARRIERS", "PILOTS_PER_USER", "SUPPORTED_GROUP_COUNTS",
    "TOTAL_FRAME_ENERGY_PER_USER", "TOTAL_PILOT_RE", "TOTAL_RE",
    "TRAIN_FEEDBACK_GAMMA_DB", "TWCv1K8Config", "all_groupings",
    "PILOT_ENERGY_CANDIDATES", "KEY_SHARED_RATIO_CANDIDATES", "candidate_energy_splits",
    "default_hybrid_layout", "enumerate_partitions", "feedback_noise_variance",
    "protocol_metadata",
]
