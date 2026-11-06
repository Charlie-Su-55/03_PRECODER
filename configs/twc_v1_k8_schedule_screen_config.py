"""Standard-library plan for the bounded K8 condition-order screening batch.

No model, checkpoint or channel is loaded by importing this module.  The old
warmup recipe and random addresses are retained; only the group schedule is
varied in an independent continuation namespace.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "twc_v1_k8_schedule_screen_v1"
OUTPUT_ROOT = REPO_ROOT / "runs" / NAMESPACE
CHECKPOINT_FORMAT = NAMESPACE
ARMS = ("blocked_universal", "interleaved_universal", "specialist_g4", "fdma_control")
SEED = 20261107
SOURCE_UPDATES = 10_000
UPDATES = 3_024
CYCLE_UPDATES = 108
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
SAVE_EVERY = 108
PROGRESS_VALIDATION_EVERY = 756
PROGRESS_VALIDATION_BATCHES = 4
GAMMAS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
PILOT_ENERGIES = (8.0, 16.0, 24.0)
ENERGY_RATIOS = ((1, 3), (1, 1))
GROUP_COUNTS = (1, 2, 4)
CDL_MODEL = "C"
SPEED_MPS = 1.0
BASE_DATA_SEED = 20261106
DATA_SEED_NAMESPACE = "twc_v1_k8_codec_warmup_v1"
DOWNLINK_SNR_DB = 25.0
VALIDATION_SAMPLES = 256
VALIDATION_BATCHES = 64
VALIDATION_START_INDEX = 1024
VALIDATION_PILOT_ENERGY = 16.0
CE_CHECKPOINT = "runs/twc_v1_k8_ce/seed_20261105/checkpoint_latest.pt"
CE_CALIBRATION = "runs/twc_v1_k8_ce/seed_20261105/calibration.csv"
CE_SHA256 = {
    "checkpoint": "0a5dc2e445e1d127252a6f0bfea488258a59e89fe767c9ee7dad99c9c488103f",
    "calibration": "d1a5dbbeb47220498e5fca25fa70e760d7b577eb9a0df7b88605f759f810d3b5",
}
SOURCE_SHA256 = {
    "fdma": "c0636e91fda1dd0118719f55665bce1841cfb72d9219354258174405c251b208",
    "hybrid": "65e02758cab40908b85973a78012d9d0adb7e4ebe04a578348679ceab4ec79c0",
}


def _arm(arm):
    if not isinstance(arm, str) or arm not in ARMS:
        raise ValueError("One of the four prescribed screening arms is required")
    return arm


def normalize_json(value):
    """Canonical JSON value, including tuple/list equivalence, rejecting NaNs."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False,
                                separators=(",", ":")))


def source_identity(arm):
    family = "fdma" if _arm(arm) == "fdma_control" else "hybrid"
    directory = REPO_ROOT / "runs/twc_v1_k8_codec_warmup" / family / f"seed_{SEED}_jointv1"
    return dict(arm=family, seed=SEED, step=SOURCE_UPDATES,
                path=str(directory / "checkpoint_latest.pt"),
                sha256=SOURCE_SHA256[family],
                manifest_path=str(directory / "manifest.json"))


def schedule(arm, local_step):
    """Address a new update without resetting the old 10k condition phase."""
    _arm(arm)
    if type(local_step) is not int or not 1 <= local_step <= UPDATES:
        raise ValueError("Local update must be an integer in [1,3024]")
    n = SOURCE_UPDATES + local_step - 1
    if arm == "blocked_universal":
        groups = GROUP_COUNTS[(n // 18) % 3]
    elif arm == "interleaved_universal":
        # A naïve n%3 would lock G to the gamma period of six.  This pattern
        # interleaves G while retaining every joint condition once per108.
        groups = GROUP_COUNTS[(n + n // 18) % 3]
    else:
        groups = 4 if arm == "specialist_g4" else 1
    return dict(local_step=local_step, global_step=n + 1,
                gamma_db=GAMMAS[n % 6],
                pilot_energy_per_ue=PILOT_ENERGIES[(n // 6) % 3],
                group_count=groups,
                energy_ratio=None if arm == "fdma_control" else
                             list(ENERGY_RATIOS[(n // 2) % 2]))


def schedule_counts(arm):
    counts = Counter()
    for step in range(1, UPDATES + 1):
        point = schedule(arm, step)
        counts[(point["group_count"], point["gamma_db"], point["pilot_energy_per_ue"],
                tuple(point["energy_ratio"] or ()))] += 1
    return [dict(group_count=g, gamma_db=gamma, pilot_energy_per_ue=ep,
                 energy_ratio=list(ratio) if ratio else None, count=count)
            for (g, gamma, ep, ratio), count in sorted(counts.items())]


def training_noise_seed(local_step):
    # Validate through the common schedule; this seed is independent of arm/G.
    step = schedule("blocked_universal", local_step)["global_step"]
    return SEED * 10_000_000 + step


def evaluation_noise_seed(batch_index, gamma_index, kind):
    """One pilot and data base tensor shared across arm/G/energy conditions."""
    if type(batch_index) is not int or not 0 <= batch_index < VALIDATION_BATCHES:
        raise ValueError("Validation batch index must be in [0,63]")
    if type(gamma_index) is not int or not 0 <= gamma_index < len(GAMMAS):
        raise ValueError("Validation gamma index must be in [0,5]")
    if kind not in ("pilot", "data"):
        raise ValueError("Noise kind must be pilot or data")
    return (SEED * 10_000_000 + 9_000_000 + batch_index * 1000 +
            gamma_index * 10 + int(kind == "data"))


def recipe(arm):
    _arm(arm)
    return normalize_json(dict(
        version=NAMESPACE, arm=arm, seed=SEED, source=source_identity(arm),
        new_updates=UPDATES, source_updates=SOURCE_UPDATES, batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE, optimizer="AdamW", weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM, scheduler="constant LambdaLR",
        optimizer_initialization="fresh; no source optimizer state inherited",
        loss="mean per-user downlink CSI NMSE", loss_weight_nmse=1.0,
        loss_weight_rate=0.0, ce_frozen=True, ce_checkpoint=CE_CHECKPOINT,
        ce_calibration=CE_CALIBRATION, downlink_snr_db=DOWNLINK_SNR_DB,
        ce_sha256=CE_SHA256,
        physical=dict(K=8, Nt=32, Nsc=256, total_feedback_RE=256,
                      total_energy_per_ue=64.0,
                      noise_variance="0.25*10**(-gamma/10)"),
        channel_data=dict(base_seed=BASE_DATA_SEED, seed_namespace=DATA_SEED_NAMESPACE,
                          cdl_model=CDL_MODEL, speed=SPEED_MPS,
                          training_role="codec_train",
                          training_address="10000+local_step"),
        conditions=dict(gamma=list(GAMMAS), pilot_energy=list(PILOT_ENERGIES),
                        key_shared_ratios=ENERGY_RATIOS,
                        schedule_formula="n=10000+local_step-1; see schedule()",
                        counts=schedule_counts(arm)),
        save_every=SAVE_EVERY, progress_validation_every=PROGRESS_VALIDATION_EVERY,
        validation=dict(role="validation", first_batch=VALIDATION_START_INDEX,
                        batch_size=BATCH_SIZE, batches=VALIDATION_BATCHES,
                        samples=VALIDATION_SAMPLES, pilot_energy=VALIDATION_PILOT_ENERGY,
                        gamma=GAMMAS, key_shared_ratios=ENERGY_RATIOS,
                        same_channels_and_pilot_data_noise=True),
        selection="terminal local step3024; no test/checkpoint selection",
        training_seed_replications=1, test_used=False,
        scope="condition-order/concentration screen; not a final performance claim"))


def fingerprint(arm):
    value = json.dumps(recipe(arm), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside_root(path, root=OUTPUT_ROOT):
    target, allowed = Path(path).resolve(), Path(root).resolve()
    if target == allowed or not target.is_relative_to(allowed):
        raise ValueError("A child path inside the independent screening root is required")
    return target


def prepare_output(path, *, resume=False, root=OUTPUT_ROOT):
    target = inside_root(path, root)
    if resume:
        if not target.is_dir() or not (target / "checkpoint_latest.pt").is_file():
            raise FileNotFoundError("Resume requires an existing directory and latest checkpoint")
    else:
        target.mkdir(parents=True, exist_ok=False)
    return target


def validate_resume_metadata(payload, arm, *, code_sha256=None):
    """Strict lightweight identity check; runtime verifies tensors and states."""
    _arm(arm)
    required = {"format", "arm", "seed", "step", "cumulative_updates", "recipe",
                "config_fingerprint", "source", "codec_model", "optimizer",
                "scheduler", "rng", "data_protocol", "ce_identity", "code_sha256"}
    if not isinstance(payload, dict) or not required <= set(payload):
        raise ValueError("Incomplete screening resume metadata")
    if (payload["format"] != CHECKPOINT_FORMAT or payload["arm"] != arm or
            type(payload["seed"]) is not int or payload["seed"] != SEED):
        raise ValueError("Screening checkpoint format/arm/seed mismatch")
    step = payload["step"]
    if type(step) is not int or not 0 <= step <= UPDATES or step % SAVE_EVERY:
        raise ValueError("Local resume step must lie in [0,3024]")
    if (type(payload["cumulative_updates"]) is not int or
            payload["cumulative_updates"] != SOURCE_UPDATES + step):
        raise ValueError("Cumulative resume update count mismatch")
    for key, expected in (("recipe", recipe(arm)), ("source", source_identity(arm)),
                          ("config_fingerprint", fingerprint(arm))):
        if normalize_json(payload[key]) != normalize_json(expected):
            raise ValueError("Screening resume " + key + " mismatch")
    for name in ("codec_model", "optimizer", "scheduler", "data_protocol", "ce_identity", "code_sha256"):
        if not isinstance(payload[name], dict) or not payload[name]:
            raise ValueError("Missing complete resume mapping: " + name)
    if (not isinstance(payload["rng"], dict) or
            not {"python", "numpy", "torch_cpu", "torch_cuda"} <= set(payload["rng"])):
        raise ValueError("Complete Python/NumPy/Torch CPU/CUDA RNG required")
    if payload["scheduler"].get("last_epoch") != step:
        raise ValueError("Scheduler must retain the local update count")
    if code_sha256 is not None and payload["code_sha256"] != code_sha256:
        raise ValueError("Resume source-code hashes changed")
    return step
