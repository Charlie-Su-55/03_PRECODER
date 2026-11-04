"""TWC 1.0 K8 physical feedback budget and resource layouts.

This module is deliberately independent of the historical K4 grouped-codec
protocol. It describes RE placement only; it does not implement a learned
encoder, pilot estimator, receiver, energy controller, or trainer.
"""
from dataclasses import dataclass
from itertools import combinations
import math


PROTOCOL = "twc_v1_k8_budget256_pilot64_key4_v1"
K_USERS = 8
N_ANTENNAS = 32
N_SUBCARRIERS = 256
TOTAL_RE = 256
PILOTS_PER_USER = 8
TOTAL_PILOT_RE = K_USERS * PILOTS_PER_USER
KEY_DIM_PER_USER = 4
TOTAL_KEY_RE = K_USERS * KEY_DIM_PER_USER
TOTAL_FRAME_ENERGY_PER_USER = 64.0
DOWNLINK_SNR_DB = 25.0
TRAIN_FEEDBACK_GAMMA_DB = (0.0, 25.0)
EVALUATION_FEEDBACK_GAMMA_DB = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
SUPPORTED_GROUP_COUNTS = (1, 2, 4)
_PILOT_START = 0
_KEY_START = TOTAL_PILOT_RE
_SHARED_START = TOTAL_PILOT_RE + TOTAL_KEY_RE


def shared_dimension(group_count):
    """Return the per-group shared dimension required to fill the RE budget."""
    _validate_group_count(group_count)
    remaining = TOTAL_RE - TOTAL_PILOT_RE - TOTAL_KEY_RE
    return remaining // group_count


def _validate_group_count(group_count):
    if type(group_count) is not int or group_count not in SUPPORTED_GROUP_COUNTS:
        raise ValueError(f"group_count must be one of {SUPPORTED_GROUP_COUNTS}")


def enumerate_partitions(group_count):
    """Enumerate unlabeled equal-size partitions of users 0..7 deterministically."""
    _validate_group_count(group_count)
    group_size = K_USERS // group_count

    def recurse(remaining):
        if not remaining:
            yield ()
            return
        first = remaining[0]
        for tail in combinations(remaining[1:], group_size - 1):
            group = (first,) + tail
            selected = set(group)
            rest = tuple(user for user in remaining if user not in selected)
            for suffix in recurse(rest):
                yield (group,) + suffix

    return tuple(recurse(tuple(range(K_USERS))))


def _normalize_partition(group_count, groups):
    _validate_group_count(group_count)
    groups = tuple(tuple(group) for group in groups)
    group_size = K_USERS // group_count
    if len(groups) != group_count or any(len(group) != group_size for group in groups):
        raise ValueError("Groups must be a balanced partition for the selected group_count")
    flattened = [user for group in groups for user in group]
    if any(type(user) is not int or user < 0 or user >= K_USERS for user in flattened):
        raise ValueError("Group members must be user indices in [0,7]")
    if sorted(flattened) != list(range(K_USERS)):
        raise ValueError("Groups must include each user exactly once")
    # Group labels have no physical significance; canonicalize to make RE maps stable.
    return tuple(sorted(tuple(sorted(group)) for group in groups))


@dataclass(frozen=True)
class HybridLayout:
    """A K8 Hybrid layout with dedicated pilots/keys and group-shared REs."""
    groups: tuple

    def __post_init__(self):
        normalized = _normalize_partition(len(self.groups), self.groups)
        object.__setattr__(self, "groups", normalized)
        if self.total_occupied_re != TOTAL_RE:
            raise ValueError("Hybrid RE budget does not equal 256")
        occupancy = [sum(re in self.resource_positions[user] for user in range(K_USERS))
                     for re in range(TOTAL_RE)]
        if tuple(occupancy[:TOTAL_PILOT_RE]) != (1,) * TOTAL_PILOT_RE:
            raise ValueError("Each pilot RE must be assigned to exactly one UE")
        if tuple(occupancy[_KEY_START:_SHARED_START]) != (1,) * TOTAL_KEY_RE:
            raise ValueError("Each key RE must be assigned to exactly one UE")
        expected_shared_occupancy = K_USERS // self.group_count
        if tuple(occupancy[_SHARED_START:]) != (expected_shared_occupancy,) * (TOTAL_RE - _SHARED_START):
            raise ValueError("Shared RE occupancy must equal the group size")

    @property
    def group_count(self):
        return len(self.groups)

    @property
    def users_per_group(self):
        return K_USERS // self.group_count

    @property
    def shared_dimension(self):
        return shared_dimension(self.group_count)

    @property
    def per_user_data_coordinates(self):
        return KEY_DIM_PER_USER + self.shared_dimension

    @property
    def total_occupied_re(self):
        return TOTAL_PILOT_RE + TOTAL_KEY_RE + self.group_count * self.shared_dimension

    @property
    def pilot_positions(self):
        return tuple(tuple(range(_PILOT_START + user * PILOTS_PER_USER,
                                 _PILOT_START + (user + 1) * PILOTS_PER_USER))
                     for user in range(K_USERS))

    @property
    def key_positions(self):
        return tuple(tuple(range(_KEY_START + user * KEY_DIM_PER_USER,
                                 _KEY_START + (user + 1) * KEY_DIM_PER_USER))
                     for user in range(K_USERS))

    @property
    def shared_positions_by_group(self):
        dimension = self.shared_dimension
        return tuple(tuple(range(_SHARED_START + group_index * dimension,
                                 _SHARED_START + (group_index + 1) * dimension))
                     for group_index in range(self.group_count))

    @property
    def resource_positions(self):
        positions = [set() for _ in range(K_USERS)]
        for user in range(K_USERS):
            positions[user].update(self.pilot_positions[user])
            positions[user].update(self.key_positions[user])
        for members, resources in zip(self.groups, self.shared_positions_by_group):
            for user in members:
                positions[user].update(resources)
        return tuple(tuple(sorted(resources)) for resources in positions)

    def to_dict(self):
        return {
            "protocol": PROTOCOL,
            "layout": "hybrid",
            "users": K_USERS,
            "antennas": N_ANTENNAS,
            "subcarriers": N_SUBCARRIERS,
            "total_feedback_re": TOTAL_RE,
            "pilot_re_total": TOTAL_PILOT_RE,
            "pilots_per_user": PILOTS_PER_USER,
            "key_re_total": TOTAL_KEY_RE,
            "key_dim_per_user": KEY_DIM_PER_USER,
            "group_count": self.group_count,
            "users_per_group": self.users_per_group,
            "groups": [list(group) for group in self.groups],
            "shared_dimension_per_group": self.shared_dimension,
            "data_coordinates_per_user": self.per_user_data_coordinates,
            "pilot_positions_by_user": [list(row) for row in self.pilot_positions],
            "key_positions_by_user": [list(row) for row in self.key_positions],
            "shared_positions_by_group": [list(row) for row in self.shared_positions_by_group],
            "user_resource_positions": [list(row) for row in self.resource_positions],
            "occupancy_rule": "pilot/key RE exclusive; shared RE reused within one group",
            "total_occupied_re_check": self.total_occupied_re,
        }


@dataclass(frozen=True)
class FDMALayout:
    """Matched orthogonal-data control: pilot64 + 8 users * 24 data RE."""

    @property
    def data_coordinates_per_user(self):
        return (TOTAL_RE - TOTAL_PILOT_RE) // K_USERS

    @property
    def pilot_positions(self):
        return tuple(tuple(range(user * PILOTS_PER_USER, (user + 1) * PILOTS_PER_USER))
                     for user in range(K_USERS))

    @property
    def data_positions(self):
        width = self.data_coordinates_per_user
        start = TOTAL_PILOT_RE
        return tuple(tuple(range(start + user * width, start + (user + 1) * width))
                     for user in range(K_USERS))

    @property
    def total_occupied_re(self):
        return TOTAL_PILOT_RE + K_USERS * self.data_coordinates_per_user

    def to_dict(self):
        return {
            "protocol": PROTOCOL,
            "layout": "fdma_control",
            "users": K_USERS,
            "antennas": N_ANTENNAS,
            "subcarriers": N_SUBCARRIERS,
            "total_feedback_re": TOTAL_RE,
            "pilot_re_total": TOTAL_PILOT_RE,
            "pilots_per_user": PILOTS_PER_USER,
            "key_dim_per_user": 0,
            "data_coordinates_per_user": self.data_coordinates_per_user,
            "pilot_positions_by_user": [list(row) for row in self.pilot_positions],
            "data_positions_by_user": [list(row) for row in self.data_positions],
            "total_occupied_re_check": self.total_occupied_re,
        }


def feedback_noise_variance(gamma_db):
    """Absolute complex-noise variance from the fixed per-UE reference energy/B."""
    if isinstance(gamma_db, bool) or not isinstance(gamma_db, (int, float)):
        raise TypeError("feedback gamma must be a real scalar")
    gamma_db = float(gamma_db)
    if not math.isfinite(gamma_db) or not TRAIN_FEEDBACK_GAMMA_DB[0] <= gamma_db <= TRAIN_FEEDBACK_GAMMA_DB[1]:
        raise ValueError("feedback gamma must be within [0,25] dB")
    return (TOTAL_FRAME_ENERGY_PER_USER / TOTAL_RE) * 10.0 ** (-gamma_db / 10.0)


def default_hybrid_layout(group_count):
    """Convenience fixed partition for smoke-free configuration serialization."""
    _validate_group_count(group_count)
    group_size = K_USERS // group_count
    groups = tuple(tuple(range(start, start + group_size))
                   for start in range(0, K_USERS, group_size))
    return HybridLayout(groups)


def protocol_metadata():
    return {
        "protocol": PROTOCOL,
        "users": K_USERS,
        "antennas": N_ANTENNAS,
        "subcarriers": N_SUBCARRIERS,
        "total_feedback_re": TOTAL_RE,
        "pilot_re_total": TOTAL_PILOT_RE,
        "pilots_per_user": PILOTS_PER_USER,
        "key_dim_per_user": KEY_DIM_PER_USER,
        "supported_group_counts": list(SUPPORTED_GROUP_COUNTS),
        "shared_dimension_per_group": {str(g): shared_dimension(g) for g in SUPPORTED_GROUP_COUNTS},
        "fdma_data_coordinates_per_user": FDMALayout().data_coordinates_per_user,
        "frame_energy_per_user": TOTAL_FRAME_ENERGY_PER_USER,
        "downlink_snr_db": DOWNLINK_SNR_DB,
        "feedback_gamma_train_range_db": list(TRAIN_FEEDBACK_GAMMA_DB),
        "feedback_gamma_evaluation_db": list(EVALUATION_FEEDBACK_GAMMA_DB),
        "N0_definition": "(64/256)*10**(-feedback_gamma_db/10)",
        "energy_split_candidates": "not configured; Ep/Ekey/EAS positive and sum to 64 per UE",
        "train_validation_test_implementation": "not implemented in this protocol-only module",
    }
