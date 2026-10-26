"""Opt-in two-seed Swin continuation control; legacy baseline defaults untouched."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
import json
from pathlib import Path

from .baseline_config import BaselineConfig
from .as_receiver_train_config import channel_data_api, make_channel_data
from evaluate_proposal_as_direct import archived_swin

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "runs_as_screen" / "swin_adaptation_v1"
SEEDS = (20261030, 20261031)


def optimizer_group_spec(cfg=None):
    return [dict(name="encoder", prefix="encoder.", peak_lr=2e-5),
            dict(name="decoder", prefix="decoder.", peak_lr=1e-4)]


@dataclass
class SwinAdaptConfig(BaselineConfig):
    source_identity: dict = field(default_factory=dict)
    channel_data_protocol: dict = field(default_factory=dict)

    @property
    def arm(self):
        return "swin_adapt"

    @property
    def EXP_NAME(self):
        return f"swin_adapt20k_K4_Nsc256_Dfb256_Df64_sharedUE_seed{self.SEED}"

    @property
    def CONFIG_PATH(self):
        return str(Path(self.LOG_DIR) / "config.json")

    @property
    def DOWNLINK_SNR_DB(self):
        return 25.0

    def to_dict(self):
        result = super().to_dict()
        result.update(
            source_identity=deepcopy(self.source_identity),
            channel_data_protocol=deepcopy(self.channel_data_protocol),
            optimizer_groups=optimizer_group_spec(self),
            adaptation_protocol=dict(
                version="swin-adaptation-v1", screening_not_formal_result=True,
                arm=self.arm, seeds=list(SEEDS), new_updates=20000,
                architecture="unchanged models.swin_precoder.Swin_E2E_Precoder",
                downlink_snr_db=25.0, feedback_snr_uniform=[0.0,25.0],
                validation_feedback_snr=[0,5,10,15,20,25],
                selection="mean_all: arithmetic average over six FB points at fixed DL25",
                source_step_zero_competes=True, all_parameters_trainable=True,
                objective="existing normalized RZF(H_hat) sum rate only",
                feedback_rng="baseline_train existing seed+40000000+step temporary seed",
                feedback_snr_rng="baseline_train existing numpy.default_rng(seed+20000000+step)",
                full_training_bitwise_resume_verified=False,
            ),
        )
        return json.loads(json.dumps(result))


def swin_adapt_config(seed, output_root=None):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"Swin adaptation seed must be one of {SEEDS}")
    source = archived_swin()
    saved = source["loaded_checkpoint_config"]
    values = {item.name: deepcopy(saved[item.name]) for item in fields(BaselineConfig)}
    values.update(OUTPUT_ROOT=str(OUTPUT_ROOT if output_root is None else Path(output_root).absolute()),
        SEED=seed, TOTAL_STEPS=20000, BATCH_SIZE=48, VAL_BATCH=64, VAL_INTERVAL=500,
        LOG_INTERVAL=100, DATA_CACHE_SIZE=1, LR=1e-4, WEIGHT_DECAY=1e-5,
        WARMUP_PCT=.1, GRAD_CLIP=.5, SNR_STAGE1_END=6000, SNR_STAGE2_END=14000,
        STAGE1_SNR=(0.,25.), STAGE2_SNR=(0.,25.), STAGE3_SNR=(0.,25.),
        VAL_SNR_LIST=(0,5,10,15,20,25), OBJECTIVE="task")
    cfg = SwinAdaptConfig(**values, source_identity=deepcopy(source))
    cfg.channel_data_protocol = make_channel_data(cfg).protocol
    return cfg


def validate_config(cfg):
    if not isinstance(cfg, SwinAdaptConfig):
        raise ValueError("Expected independent SwinAdaptConfig")
    if cfg.to_dict() != swin_adapt_config(cfg.SEED, cfg.OUTPUT_ROOT).to_dict():
        raise ValueError("Fixed Swin adaptation source/architecture/recipe/protocol mismatch")
    channel_data_api().validate_channel_data(cfg, make_channel_data(cfg))
