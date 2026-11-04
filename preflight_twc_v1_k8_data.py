#!/usr/bin/env python3
"""CPU-only real-CDL replay, role-isolation and resume-address preflight."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
_DATA_NAME = "_twc_v1_k8_replayable_data_preflight"
_DATA_PATH = ROOT / "utils" / "twc_v1_k8_replayable_data.py"
_SPEC = importlib.util.spec_from_file_location(_DATA_NAME, _DATA_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Cannot load K8 replayable data module")
_DATA_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_DATA_NAME] = _DATA_MODULE
_SPEC.loader.exec_module(_DATA_MODULE)
ROLE_TAGS = _DATA_MODULE.ROLE_TAGS
TWCv1K8ReplayableData = _DATA_MODULE.TWCv1K8ReplayableData
validate_disjoint_addresses = _DATA_MODULE.validate_disjoint_addresses
if "utils" in sys.modules:
    raise AssertionError("Preflight must not execute the side-effectful utils package initializer")


def main():
    torch.set_num_threads(1)
    batch_sizes = {role: 1 for role in ROLE_TAGS if role != "constructor"}
    data = TWCv1K8ReplayableData(base_seed=20261104, batch_sizes=batch_sizes,
                                 device="cpu", cdl_model="C")
    protocol = data.protocol
    if protocol["physical"]["k_users"] != 8 or protocol["execution"]["device"] != "cpu":
        raise AssertionError("K8/CPU protocol identity mismatch")
    if len({data.address(role, 0 if role not in ("ce_train", "codec_train") else 1)["sionna_seed"]
            for role in ROLE_TAGS}) != len(ROLE_TAGS):
        raise AssertionError("K8 role-derived Sionna seeds are not unique")

    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    roles = ("ce_train", "codec_train", "ce_calibration", "validation", "test")
    samples = {}
    for role in roles:
        index = 1 if role in ("ce_train", "codec_train") else 0
        samples[role] = data.sample(role, index)
    validation_again = data.sample("validation", 0)
    ce_train_resume = data.sample("ce_train", 1)
    if random.getstate() != py_before:
        raise AssertionError("Sionna sampling changed Python global RNG state")
    np_after = np.random.get_state()
    if (np_before[0] != np_after[0] or not np.array_equal(np_before[1], np_after[1])
            or np_before[2:] != np_after[2:]):
        raise AssertionError("Sionna sampling changed NumPy global RNG state")
    if not torch.equal(torch_before, torch.get_rng_state()):
        raise AssertionError("Sionna sampling changed Torch CPU global RNG state")
    if samples["validation"]["sample_fingerprint"] != validation_again["sample_fingerprint"]:
        raise AssertionError("Fixed validation sample is not reproducible")
    if samples["ce_train"]["sample_fingerprint"] != ce_train_resume["sample_fingerprint"]:
        raise AssertionError("Step-addressed CE training sample is not resume-reproducible")

    records = [dict(role=role, index=1 if role in ("ce_train", "codec_train") else 0,
                    sample_fingerprint=samples[role]["sample_fingerprint"])
               for role in roles]
    isolation = validate_disjoint_addresses(data, records)
    for role, sample in samples.items():
        for name in ("H_dl", "H_ul"):
            tensor = sample[name]
            if tuple(tensor.shape) != (1, 8, 32, 256) or tensor.dtype != torch.complex64:
                raise AssertionError(f"Invalid {role}/{name} K8 channel geometry")
    if len({samples[role]["sample_fingerprint"] for role in roles}) != len(roles):
        raise AssertionError("Different K8 roles unexpectedly reused a channel pair")

    print(json.dumps(dict(
        status="K8_REAL_CDL_CPU_DATA_PREFLIGHT_PASS",
        device="cpu",
        sionna_version=protocol["sionna_version"],
        torch_version=str(torch.__version__),
        data_id=protocol["data_id"],
        channel_geometry=[1, 8, 32, 256],
        cdl_profile=protocol["physical"]["cdl_model"],
        roles=list(roles),
        samples={role: {"address": sample["address"],
                        "sample_fingerprint": sample["sample_fingerprint"],
                        "H_dl_shape": list(sample["H_dl"].shape),
                        "H_ul_shape": list(sample["H_ul"].shape)}
                 for role, sample in samples.items()},
        rng_restoration="Python/NumPy/Torch CPU global states unchanged",
        fixed_validation_replay="pass",
        training_step_pause_resume_replay="pass",
        isolation=isolation,
        optimizer_updates=0,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
