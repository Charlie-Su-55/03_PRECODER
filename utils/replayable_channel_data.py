"""Opt-in, step-addressed CDL data; no heavy imports or GPU work on construction."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "training-channel-data-v1"
CONFIG_KEY = "channel_data_protocol"
NAMESPACE_BITS = 30
NAMESPACE_SIZE = 1 << NAMESPACE_BITS
MAX_TRAIN_STEP = NAMESPACE_SIZE - 1
NAMESPACES = {"constructor": 0, "validation": 1, "training": 2}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _integer(value, name, minimum=1, maximum=MAX_TRAIN_STEP):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _device_name(value):
    value = str(value)
    if value == "cuda":
        value = "cuda:0"
    if value != "cpu" and not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value):
        raise ValueError("Data device must be cpu or an explicit cuda:N device")
    return value


def _load_source(relative, name):
    """Load genuine local sources without the legacy utils package initializer."""
    path = ROOT / relative
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError(f"Conflicting module identity: {name}")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class ReplayableChannelData:
    """One complete DL/UL pair per address; no per-batch hashes or CPU copies.

    Constructor is metadata-only. prepare() isolates real generator construction;
    validation() and training(step) isolate sampling. Cached generator objects
    are not serialized. A validation call regenerates the same fixed data.
    """

    def __init__(self, *, base_seed, k_users=4, nt=32, nsc=256, batch_size=2,
                 validation_batch_size=2, carrier_freq=3.5e9, speed=1.0,
                 cdl_model="C", device="cuda:0"):
        base_seed = _integer(base_seed, "base_seed", 0, 2**32 - 1)
        for name, value in (("k_users", k_users), ("nt", nt), ("nsc", nsc),
                            ("batch_size", batch_size), ("validation_batch_size", validation_batch_size)):
            _integer(value, name)
        if nt != 32:
            raise ValueError("Unmodified channel generator uses a fixed 4x8 BS array (Nt=32)")
        if (isinstance(carrier_freq, bool) or not isinstance(carrier_freq, (int, float))
                or not math.isfinite(carrier_freq) or carrier_freq <= 0):
            raise ValueError("carrier_freq must be finite and positive")
        if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                or not math.isfinite(speed) or speed < 0):
            raise ValueError("speed must be finite and nonnegative")
        if cdl_model not in ("A", "B", "C", "D", "E"):
            raise ValueError("cdl_model must be one of A/B/C/D/E")
        if NAMESPACES != {"constructor": 0, "validation": 1, "training": 2}:
            raise ValueError("Conflicting or unsupported seed namespace registry")
        identity = dict(protocol_version=PROTOCOL_VERSION, sionna_version="2.0.1",
                        rng_isolation="sionna-rng-isolation-v1", base_seed=base_seed,
                        physical=dict(k_users=k_users, nt=nt, nsc=nsc,
                                      carrier_freq=float(carrier_freq), speed=float(speed), cdl_model=cdl_model,
                                      delay_spread=300e-9, subcarrier_spacing=30000.0,
                                      bs_array=[4, 8], ue_array=[1, 1], polarization="single-V",
                                      antenna_pattern="38.901", direction="downlink", num_time_steps=1,
                                      cyclic_prefix_fraction=0.07, cir_normalize=True,
                                      per_user_mean_power_normalize=True, normalization_epsilon=1e-12),
                        batch_sizes=dict(training=batch_size, validation=validation_batch_size),
                        execution=dict(device=_device_name(device), sionna_precision="single"))
        data_id = hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()
        offsets = {split: int.from_bytes(hashlib.sha256(
            (PROTOCOL_VERSION + ":" + data_id + ":" + split).encode("utf-8")).digest()[:4], "big")
                   % NAMESPACE_SIZE for split in NAMESPACES}
        protocol = dict(identity, data_id=data_id, seed_mapping=dict(
            version="namespace-offset30-v1", namespaces=dict(NAMESPACES), offsets=offsets,
            index_bits=NAMESPACE_BITS, max_training_step=MAX_TRAIN_STEP,
            formula="(namespace << 30) | ((offset + index) % 2**30)",
            constructor_index=0, validation_index=0, training_index="global_step (1-based)",
            offset_derivation="SHA256(UTF8(protocol_version + ':' + data_id + ':' + split))[:4], big-endian, mod 2**30",
            data_id_derivation="SHA256(canonical identity JSON: sorted keys, compact separators, ensure_ascii=True, UTF8)"))
        self._protocol_json = _canonical(protocol)
        self._generator = None
        self._rng = None
        self._torch = None

    @property
    def protocol(self):
        return json.loads(self._protocol_json)

    def seed_for(self, split, index=0):
        mapping = self.protocol["seed_mapping"]
        if split not in mapping["namespaces"]:
            raise ValueError(f"Unknown seed namespace: {split}")
        if split == "training":
            _integer(index, "global_step")
        elif type(index) is not int or index != 0:
            raise ValueError("Constructor and fixed validation keys require index=0")
        seed = (mapping["namespaces"][split] << NAMESPACE_BITS) | ((mapping["offsets"][split] + index) % NAMESPACE_SIZE)
        _integer(seed, "derived seed", 0, 2**32 - 1)
        return seed

    def prepare(self):
        if self._generator is not None:
            return self
        rng = _load_source("utils/sionna_rng.py", "_precoder_sionna_rng_isolation_v1")
        torch, _, config = rng.runtime()
        p = self.protocol
        if config.precision != p["execution"]["sionna_precision"]:
            raise RuntimeError("Sionna precision differs from declared data protocol; no silent override")
        with rng.isolated_sionna_seed(self.seed_for("constructor")), torch.no_grad():
            source = _load_source("utils/channel_generator.py", "_precoder_replayable_cdl_source_v1")
            physical = p["physical"]
            generator = source.CDLChannelGenerator_Feedback(
                Nt=physical["nt"], Nsc=physical["nsc"], carrier_freq=physical["carrier_freq"],
                speed=physical["speed"], num_users=physical["k_users"], cdl_model=physical["cdl_model"]
            ).to(p["execution"]["device"])
        self._rng, self._torch, self._generator = rng, torch, generator
        return self

    def _sample(self, split, index):
        seed = self.seed_for(split, index)
        self.prepare()
        p = self.protocol
        with self._rng.isolated_sionna_seed(seed) as config, self._torch.no_grad():
            if config.precision != p["execution"]["sionna_precision"]:
                raise RuntimeError("Sionna precision changed since data preparation")
            # The unchanged public method samples DL then UL with no intervening reset.
            return self._generator.generate_batch_data(
                p["batch_sizes"][split], device=p["execution"]["device"])

    def validation(self):
        return self._sample("validation", 0)

    def training(self, global_step):
        return self._sample("training", global_step)


def validate_protocol(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("Explicit channel_data_protocol must be a JSON mapping")
    try:
        physical, batches = protocol["physical"], protocol["batch_sizes"]
        rebuilt = ReplayableChannelData(
            base_seed=protocol["base_seed"], k_users=physical["k_users"], nt=physical["nt"],
            nsc=physical["nsc"], carrier_freq=physical["carrier_freq"], speed=physical["speed"],
            cdl_model=physical["cdl_model"], batch_size=batches["training"],
            validation_batch_size=batches["validation"], device=protocol["execution"]["device"])
        if _canonical(protocol) != _canonical(rebuilt.protocol):
            raise ValueError("Data protocol identity/mapping/version differs from supported v1")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete or invalid channel data protocol") from exc
    return protocol


def validate_channel_data(cfg, channel_data, *, device=None):
    """Pure validation; call before trainer side effects or output creation."""
    config = cfg.to_dict()
    if not isinstance(config, dict):
        raise ValueError("cfg.to_dict() must return a mapping")
    declared = CONFIG_KEY in config
    if not declared and channel_data is None:
        return None  # Preserve legacy config serialization, RNG and data behavior.
    if not declared or channel_data is None:
        raise ValueError("New channel data requires BOTH explicit cfg.to_dict() declaration and data object")
    protocol = validate_protocol(config[CONFIG_KEY])
    if any(not callable(getattr(channel_data, name, None)) for name in ("prepare", "validation", "training")):
        raise ValueError("channel_data must implement prepare(), validation(), training(global_step)")
    if _canonical(protocol) != _canonical(getattr(channel_data, "protocol", None)):
        raise ValueError("Config and channel_data protocol mismatch")
    expected = {"K_USERS": protocol["physical"]["k_users"], "ANTENNAS": protocol["physical"]["nt"],
                "SUBCARRIERS": protocol["physical"]["nsc"], "CARRIER_FREQ": protocol["physical"]["carrier_freq"],
                "SPEED": protocol["physical"]["speed"], "BATCH_SIZE": protocol["batch_sizes"]["training"],
                "VAL_BATCH": protocol["batch_sizes"]["validation"]}
    for name, value in expected.items():
        if getattr(cfg, name, None) != value:
            raise ValueError(f"Data protocol/config {name} mismatch: expected {value}, found {getattr(cfg, name, None)}")
    _integer(cfg.TOTAL_STEPS, "TOTAL_STEPS")
    if device is not None and _device_name(device) != protocol["execution"]["device"]:
        raise ValueError("Trainer device differs from explicitly declared data device")
    return protocol


def configuration_fingerprint(config):
    """Same JSON spacing/encoding as main_train for JSON-native new configs."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()[:16]


def validate_resume_config(checkpoint, cfg):
    """New-protocol resume cannot use legacy missing-fingerprint fallback."""
    config = cfg.to_dict()
    if CONFIG_KEY not in config:
        raise ValueError("Strict data resume requires explicit protocol declaration")
    validate_protocol(config[CONFIG_KEY])
    required = {"config", "config_fingerprint", "state", "optimizer", "scheduler", "rng_state", "step", "best_metric"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError("New-protocol checkpoint lacks full config/fingerprint or resume fields")
    if _canonical(checkpoint["config"]) != _canonical(config):
        raise ValueError("Full checkpoint configuration differs from declared replayable-data run")
    if checkpoint["config_fingerprint"] != configuration_fingerprint(config):
        raise ValueError("Missing or mismatched replayable-data config fingerprint")
    _integer(checkpoint["step"], "checkpoint step", maximum=cfg.TOTAL_STEPS)
    for key in ("state", "optimizer", "scheduler", "rng_state"):
        if not isinstance(checkpoint[key], dict) or not checkpoint[key]:
            raise ValueError(f"Missing {key} resume state")
    if not isinstance(checkpoint["best_metric"], (int, float)) or not math.isfinite(checkpoint["best_metric"]):
        raise ValueError("Invalid selected validation metric")
    return checkpoint
