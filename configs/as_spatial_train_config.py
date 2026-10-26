"""Isolated AS-spatial screening config; no legacy registry/default changes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from .train_config import (BaselineDefaults, ExperimentSpec, PhysicalDefaults,
                           ProposalModelDefaults, TrainConfig, TrainingRecipe)
from .as_receiver_train_config import channel_data_api, make_channel_data
from preflight_proposal_as_direct import load_source_record

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "runs_as_screen" / "proposal_as_spatial_v1"
ARMS = ("frozen_ue", "joint")
SEEDS = (20261030, 20261031)
BS_PEAK_LR = 1e-4
UE_PEAK_LR = 2e-5
DECODER_SETTINGS = {
    "shared_across_users": True, "ordered_complex_symbols": 256,
    "input": "ordered per-symbol real/imag/reliability triples plus SNR/25",
    "input_features": 769, "spatial_expansion": [769, 8192],
    "patch_grid": [8, 8], "angular_delay_shape": [32, 32],
    "patch_size": 4, "width": 128, "heads": 4, "window_size": 4,
    "swin_blocks": 4, "mlp_ratio": 4.0, "dropout": 0.0,
    "shifted_windows": False,
    "block_implementation": "models.swin_precoder.SwinTransformerBlock",
    "initialization": "new random BS; no imported Full/Swin BS weights",
    "inverse_transform": "models.baseline_common.inverse_angular_delay_transform",
}
RECEIVER_SETTINGS = {
    "function": "adaptive_hybrid_as_receiver.regularized_recovery",
    "lambda_floor": 1e-6, "solve_dtype": "complex64",
    "noise_estimator": "mean(abs(Y_ac)**2)/(1+10**(feedback_snr_db/10))",
    "noise_reduction": "all_batch_antennas_active_AS_symbols",
    "lambda_rule": "max(noise_est,lambda_floor)",
    "physical_slice": "K*d_f:K*d_f+d_a",
}
RECIPE = TrainingRecipe(
    name="as_spatial_screen20k", total_steps=20000, batch_size=48, lr=BS_PEAK_LR,
    val_interval=500, val_batch=64, warmup_pct=0.10, grad_clip=0.5,
    phase1_pct=0.4, snr_stage1_pct=0.3, snr_stage2_pct=0.7,
    stage1_snr=(0.0, 25.0), stage2_snr=(0.0, 25.0), stage3_snr=(0.0, 25.0),
    val_snr_list=(0, 5, 10, 15, 20, 25), downlink_snr_db=25.0,
    ph1_w_dir=0.0, ph1_w_mse=0.0, ph2_w_dir=0.0, ph2_w_mse=0.0,
    weight_decay=1e-5, data_cache_size=1, log_interval=100, best_metric="mean_all",
)


def optimizer_group_spec(cfg):
    result = [dict(name="bs", peak_lr=BS_PEAK_LR)]
    if cfg.arm == "joint":
        result.append(dict(name="ue", peak_lr=UE_PEAK_LR))
    return result


@dataclass
class ASSpatialTrainConfig(TrainConfig):
    arm: str = "frozen_ue"
    source_identity: dict = field(default_factory=dict)
    channel_data_protocol: dict = field(default_factory=dict)

    @property
    def AS_SPATIAL_LAMBDA_FLOOR(self):
        return 1e-6

    @property
    def AS_SPATIAL_SOLVE_DTYPE(self):
        return "complex64"

    @property
    def architecture_metadata(self):
        return {
            "architecture": "as_spatial_v1", "model_class": "AdaptiveHybridPrecoderASSpatial",
            "arm": self.arm, "ue_ffn_dim": 2048, "D_MODEL": 256,
            "max_dimensions": [64, 256], "allocation_grid": [[0, 256]],
            "running_pwr_shape": [1], "cfg_idx": 0,
            "ue_source_components": ["encoder", "cond_embed", "running_pwr"],
            "bs_initialization": "seeded_random_same_between_arms",
            "ue_frozen": self.arm == "frozen_ue",
            "lambda_floor": self.AS_SPATIAL_LAMBDA_FLOOR,
            "solve_dtype": self.AS_SPATIAL_SOLVE_DTYPE,
            "decoder": deepcopy(DECODER_SETTINGS), "receiver": deepcopy(RECEIVER_SETTINGS),
            "running_power_ema_momentum": 0.02,
            "frozen_ue_power": "source running_pwr, no updates",
            "joint_ue_power": "original batch scale and EMA in train; frozen EMA in eval",
            "rzf_regularization": 1e-3, "power_normalization": "per-subcarrier",
        }

    def to_dict(self):
        result = super().to_dict()
        result.update(
            arm=self.arm, architecture_metadata=self.architecture_metadata,
            source_identity=deepcopy(self.source_identity),
            channel_data_protocol=deepcopy(self.channel_data_protocol),
            optimizer_groups=optimizer_group_spec(self),
            screening_protocol={
                "screening_not_formal_result": True, "new_optimizer_updates": 20000,
                "seed": self.SEED, "planned_seeds": list(SEEDS),
                "task_path": "H_hat -> normalized RZF -> sum-rate",
                "optimizer": "AdamW", "scheduler": "OneCycleLR",
                "source_bs_weights_loaded": False,
                "ue_parameter_prefixes": ["encoder.", "cond_embed."],
                "frozen_arm_preserves_running_pwr_and_ue_eval": True,
                "fresh_post_factory_default_rng_seed": self.SEED,
                "resume_bitwise_training_equivalence_verified": False,
            },
        )
        return result


def as_spatial_config(arm, seed, output_root=None):
    if arm not in ARMS:
        raise ValueError(f"Unknown AS-spatial arm: {arm!r}")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"AS-spatial seed must be one of {SEEDS}")
    source = load_source_record()
    saved = source["loaded_checkpoint_config"]
    cfg = ASSpatialTrainConfig(
        spec=ExperimentSpec(model_name="proposal", train_mode="specialist", k_users=4,
                            subcarriers=256, feedback_budget=256, recipe_name=RECIPE.name,
                            seed=seed, allocation=(0, 256), variant=arm),
        allocations=((0, 256),),
        full_allocation_grid=tuple(tuple(a) for a in saved["full_allocation_grid"]),
        recipe=RECIPE, physical=PhysicalDefaults(**saved["physical"]),
        proposal=ProposalModelDefaults(**saved["proposal_model"]),
        baselines=BaselineDefaults(**saved["baseline_model"]),
        output_root=OUTPUT_ROOT if output_root is None else Path(output_root).absolute(),
        arm=arm, source_identity=deepcopy(source),
    )
    cfg.channel_data_protocol = make_channel_data(cfg).protocol
    return cfg


def validate_config(cfg):
    if not isinstance(cfg, ASSpatialTrainConfig):
        raise ValueError("Expected isolated ASSpatialTrainConfig")
    expected = as_spatial_config(cfg.arm, cfg.SEED, cfg.output_root)
    if cfg.to_dict() != expected.to_dict():
        raise ValueError("Fixed AS-spatial arm/seed/recipe/source/data protocol mismatch")
    channel_data_api().validate_channel_data(cfg, make_channel_data(cfg))
