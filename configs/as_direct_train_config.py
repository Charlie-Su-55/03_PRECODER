"""Isolated, stdlib-only two-arm screening config; no global recipe edits."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from .train_config import (
    BaselineDefaults, ExperimentSpec, PhysicalDefaults, ProposalModelDefaults,
    TrainConfig, TrainingRecipe,
)
from preflight_proposal_as_direct import load_source_record

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "runs_as_screen" / "proposal_as_direct_v1"
ARMS = ("full_control", "as_direct")
RECIPE = TrainingRecipe(
    name="as_direct_screen5k", total_steps=5000, batch_size=48, lr=2e-5,
    val_interval=500, val_batch=64, warmup_pct=0.10, grad_clip=0.5,
    phase1_pct=0.4, snr_stage1_pct=0.3, snr_stage2_pct=0.7,
    stage1_snr=(0.0, 25.0), stage2_snr=(0.0, 25.0), stage3_snr=(0.0, 25.0),
    val_snr_list=(0, 5, 10, 15, 20, 25), downlink_snr_db=25.0,
    ph1_w_dir=0.0, ph1_w_mse=0.0, ph2_w_dir=0.0, ph2_w_mse=0.0,
    weight_decay=1e-5, data_cache_size=1, log_interval=100, best_metric="mean_all",
)


@dataclass
class ASDirectTrainConfig(TrainConfig):
    arm: str = "full_control"
    source_identity: dict = field(default_factory=dict)

    @property
    def architecture_metadata(self):
        direct = self.arm == "as_direct"
        return {
            "arm": self.arm,
            "model_class": "AdaptiveHybridPrecoderASDirect" if direct else "AdaptiveHybridPrecoder",
            "architecture": "as_direct_v1" if direct else "full",
            "ue_ffn_dim": 2048, "D_MODEL": 256, "ue_layers_per_branch": 4,
            "refinement_layers": 5, "additional_parameters": 5 if direct else 0,
            "max_dimensions": [64, 256], "allocation_grid": [[0, 256]],
            "running_pwr_shape": [1], "cfg_idx": 0,
        }

    def to_dict(self):
        result = super().to_dict()
        result.update(
            arm=self.arm, architecture_metadata=self.architecture_metadata,
            source_identity=deepcopy(self.source_identity),
            screening_protocol={
                "single_seed_screening_not_formal_result": True,
                "new_optimizer_updates": 5000, "seed": 20261025,
                "task_path": "H_hat -> normalized RZF -> sum-rate",
                "optimizer": "AdamW", "scheduler": "OneCycleLR",
                "all_parameters_trainable": True, "gate_lr_multiplier": 1.0,
                "validation_channel_seed": self.SEED + 500_000,
                "training_channel_seed": "seed + 10000000 + global_step",
                "validation_noise_seed": "seed + 1000000 + cfg_idx*10000 + FB_SNR*10",
            },
        )
        return result


def as_direct_config(arm, output_root=None):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm!r}")
    source = load_source_record()  # Same pinned identity and source validation as preflight.
    saved = source["loaded_checkpoint_config"]
    root = OUTPUT_ROOT if output_root is None else Path(output_root).absolute()
    return ASDirectTrainConfig(
        spec=ExperimentSpec(
            model_name="proposal", train_mode="specialist", k_users=4,
            subcarriers=256, feedback_budget=256, recipe_name=RECIPE.name,
            seed=20261025, allocation=(0, 256), variant=arm,
        ),
        allocations=((0, 256),),
        full_allocation_grid=tuple(tuple(a) for a in saved["full_allocation_grid"]),
        recipe=RECIPE,
        physical=PhysicalDefaults(**saved["physical"]),
        proposal=ProposalModelDefaults(**saved["proposal_model"]),
        baselines=BaselineDefaults(**saved["baseline_model"]),
        output_root=root, arm=arm, source_identity=deepcopy(source),
    )


def validate_config(cfg):
    """Compare the complete snapshot, not just an experiment directory name."""
    if not isinstance(cfg, ASDirectTrainConfig):
        raise ValueError("Expected ASDirectTrainConfig")
    expected = as_direct_config(cfg.arm, cfg.output_root)
    if cfg.to_dict() != expected.to_dict():
        raise ValueError("Two-arm screening config/source identity differs from the fixed protocol")
