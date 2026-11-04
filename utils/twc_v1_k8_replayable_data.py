"""Step-addressed K8 CDL data with isolated CE/codec/calibration/val/test roles.

The historical K4 replay protocol remains untouched. Sionna is loaded lazily,
and construction is metadata-only until ``prepare`` or ``sample`` is called.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "twc-v1-k8-replayable-cdl-v1"
SIONNA_VERSION = "2.0.1"
INDEX_BITS = 29
INDEX_SIZE = 1 << INDEX_BITS
MAX_INDEX = INDEX_SIZE - 1
ROLE_TAGS = {
    "constructor": 0,
    "ce_train": 1,
    "codec_train": 2,
    "ce_calibration": 3,
    "validation": 4,
    "test": 5,
}
TRAINING_ROLES = frozenset(("ce_train", "codec_train"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _load_source(relative, module_name):
    path = ROOT / relative
    if module_name in sys.modules:
        module = sys.modules[module_name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError(f"Conflicting module identity for {module_name}")
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load local source {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def _positive_int(value, name, maximum=MAX_INDEX):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return value


class TWCv1K8ReplayableData:
    """Regenerate complete independent DL/UL CDL pairs from role/index addresses.

    Training addresses use one-based global optimizer steps. Nontraining roles
    use explicit zero-based batch addresses, so fixed validation/calibration
    sets and final test sets cannot collide with either training stream.
    Channel generators are recreated from metadata whenever a process resumes;
    only the role/index address, never mutable PRNG state, defines a batch.
    """

    def __init__(self, *, base_seed, batch_sizes, device="cpu", cdl_model="C",
                 carrier_freq=3.5e9, speed=1.0, delay_spread=300e-9,
                 subcarrier_spacing=30e3, seed_namespace="twc_v1_k8_data_v1"):
        if type(base_seed) is not int or not 0 <= base_seed <= 2**32 - 1:
            raise ValueError("base_seed must be an explicit uint32 integer")
        if (not isinstance(batch_sizes, dict) or set(batch_sizes) != set(ROLE_TAGS) - {"constructor"}
                or any(type(v) is not int or v < 1 for v in batch_sizes.values())):
            raise ValueError("batch_sizes must declare positive sizes for all five data roles")
        device = str(device)
        if device == "cuda":
            device = "cuda:0"
        if device != "cpu" and not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device):
            raise ValueError("device must be cpu or an explicit cuda:N")
        cdl_model = str(cdl_model).upper()
        if cdl_model not in ("A", "B", "C", "D", "E"):
            raise ValueError("cdl_model must be one of A/B/C/D/E")
        for name, value, allow_zero in (("carrier_freq", carrier_freq, False),
                                        ("speed", speed, True),
                                        ("delay_spread", delay_spread, False),
                                        ("subcarrier_spacing", subcarrier_spacing, False)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or not allow_zero and value == 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
        if float(delay_spread) != 300e-9 or float(subcarrier_spacing) != 30e3:
            raise ValueError("The reused K8 CDL generator fixes delay_spread=300 ns and subcarrier_spacing=30 kHz")
        if not isinstance(seed_namespace, str) or not seed_namespace.strip():
            raise ValueError("seed_namespace must be a nonempty explicit string")

        identity = dict(protocol_version=PROTOCOL_VERSION, seed_namespace=seed_namespace,
            sionna_version=SIONNA_VERSION, rng_isolation="sionna-rng-isolation-v1",
            base_seed=base_seed, physical=dict(k_users=8, nt=32, nsc=256,
                carrier_freq=float(carrier_freq), speed=float(speed), cdl_model=cdl_model,
                delay_spread=float(delay_spread), subcarrier_spacing=float(subcarrier_spacing),
                direction="downlink", num_time_steps=1, bs_array=[4, 8], ue_array=[1, 1],
                single_polarized_vertical=True, antenna_pattern="38.901",
                cir_normalize=True, per_user_mean_power_normalize=True,
                normalization_epsilon=1e-12, H_dl_and_H_ul="independent snapshots from same CDL profile"),
            batch_sizes={key: batch_sizes[key] for key in sorted(batch_sizes)},
            execution=dict(device=device, sionna_precision="single"),
            roles={key: value for key, value in ROLE_TAGS.items()},
            address=dict(index_bits=INDEX_BITS, formula="(role_tag << 29) | ((offset + index) mod 2**29)",
                role_offsets="SHA256(canonical protocol identity + role) mod 2**29",
                training_index="1-based optimizer global_step",
                fixed_index="0-based batch index within calibration/validation/test role"))
        self._identity_json = _canonical(identity)
        identity_id = hashlib.sha256(self._identity_json.encode("utf-8")).hexdigest()
        offsets = {role: int.from_bytes(hashlib.sha256(
            (PROTOCOL_VERSION + ":" + identity_id + ":" + role).encode("utf-8")).digest()[:4], "big")
            % INDEX_SIZE for role in ROLE_TAGS}
        self._protocol_json = _canonical(dict(identity, data_id=identity_id, role_offsets=offsets))
        self._generator = None
        self._rng = None
        self._torch = None

    @property
    def protocol(self):
        return json.loads(self._protocol_json)

    def address(self, role, index):
        if role not in ROLE_TAGS:
            raise ValueError("A declared data role is required")
        if role == "constructor":
            if type(index) is not int or index != 0:
                raise ValueError("Constructor address is fixed at index zero")
        elif role in TRAINING_ROLES:
            index = _positive_int(index, "global_step")
        else:
            index = index if type(index) is int and 0 <= index <= MAX_INDEX else None
        if index is None:
            raise ValueError("Calibration/validation/test index must be in [0,2**29-1]")
        p = self.protocol
        seed = (ROLE_TAGS[role] << INDEX_BITS) | ((p["role_offsets"][role] + index) % INDEX_SIZE)
        return dict(protocol_version=PROTOCOL_VERSION, data_id=p["data_id"], role=role,
                    role_tag=ROLE_TAGS[role], index=index, sionna_seed=seed)

    def prepare(self):
        if self._generator is not None:
            return self
        rng = _load_source("utils/sionna_rng.py", "_twc_v1_k8_sionna_rng")
        torch, _, config = rng.runtime()
        p = self.protocol
        if config.precision != p["execution"]["sionna_precision"]:
            raise RuntimeError("Sionna precision differs from the K8 data protocol")
        with rng.isolated_sionna_seed(self.address("constructor", 0)["sionna_seed"]), torch.no_grad():
            source = _load_source("utils/channel_generator.py", "_twc_v1_k8_channel_generator")
            phy = p["physical"]
            generator = source.CDLChannelGenerator_Feedback(
                Nt=phy["nt"], Nsc=phy["nsc"], carrier_freq=phy["carrier_freq"],
                speed=phy["speed"], num_users=phy["k_users"], cdl_model=phy["cdl_model"]
            ).to(p["execution"]["device"])
        self._rng, self._torch, self._generator = rng, torch, generator
        return self

    def sample(self, role, index):
        address = self.address(role, index)
        self.prepare()
        p = self.protocol
        with self._rng.isolated_sionna_seed(address["sionna_seed"]) as config, self._torch.no_grad():
            if config.precision != p["execution"]["sionna_precision"]:
                raise RuntimeError("Sionna precision changed since K8 data preparation")
            H_dl, H_ul = self._generator.generate_batch_data(
                p["batch_sizes"][role], device=p["execution"]["device"])
        expected = (p["batch_sizes"][role], 8, 32, 256)
        if (tuple(H_dl.shape) != expected or tuple(H_ul.shape) != expected
                or H_dl.dtype != self._torch.complex64 or H_ul.dtype != self._torch.complex64
                or not bool(self._torch.isfinite(H_dl).all() & self._torch.isfinite(H_ul).all())):
            raise RuntimeError("K8 CDL generator returned invalid channel tensors")
        return dict(H_dl=H_dl, H_ul=H_ul, address=address,
                    sample_fingerprint=pair_fingerprint(H_dl, H_ul))


def pair_fingerprint(H_dl, H_ul):
    """Content hash for a generated complete multiuser DL/UL pair."""
    digest = hashlib.sha256()
    for name, value in (("H_dl", H_dl), ("H_ul", H_ul)):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_disjoint_addresses(data, records):
    """Reject duplicate role/index addresses and cross-role channel hashes."""
    if not isinstance(data, TWCv1K8ReplayableData) or not isinstance(records, (list, tuple)):
        raise TypeError("Explicit K8 replay data and a sequence of sample records are required")
    addresses, fingerprints = set(), {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"role", "index", "sample_fingerprint"}:
            raise ValueError("Each record must contain exactly role/index/sample_fingerprint")
        address = data.address(record["role"], record["index"])
        key = (address["role"], address["index"])
        if key in addresses:
            raise ValueError("Duplicate K8 sample address")
        addresses.add(key)
        previous = fingerprints.setdefault(record["sample_fingerprint"], record["role"])
        if previous != record["role"]:
            raise ValueError("Identical channel-pair fingerprint crossed data roles")
    return dict(records=len(records), distinct_addresses=len(addresses),
                distinct_roles=sorted({role for role, _ in addresses}),
                cross_role_fingerprint_collisions=0)


__all__ = ["TWCv1K8ReplayableData", "PROTOCOL_VERSION", "ROLE_TAGS",
           "pair_fingerprint", "validate_disjoint_addresses"]
