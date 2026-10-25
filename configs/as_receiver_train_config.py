"""Local, explicit two-seed ASReceiver screening recipe; legacy configs unchanged."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys

from .train_config import (BaselineDefaults, ExperimentSpec, PhysicalDefaults,
                           ProposalModelDefaults, TrainConfig, TrainingRecipe)
from preflight_proposal_as_direct import load_source_record

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "runs_as_screen" / "proposal_as_receiver_v1"
ARMS = ("full_control", "as_receiver")
SEEDS = (20261030, 20261031)
RECEIVER_SETTINGS = {
    "architecture": "as_receiver_v1", "lambda_floor": 1e-6,
    "solve_dtype": "complex64", "token_width": 128, "attention_heads": 4,
    "attention_dropout": 0.0, "output_projection_initialization": "zero",
}
RECIPE = TrainingRecipe(
    name="as_receiver_screen20k", total_steps=20000, batch_size=48, lr=2e-5,
    val_interval=500, val_batch=64, warmup_pct=0.10, grad_clip=0.5,
    phase1_pct=0.4, snr_stage1_pct=0.3, snr_stage2_pct=0.7,
    stage1_snr=(0.0, 25.0), stage2_snr=(0.0, 25.0), stage3_snr=(0.0, 25.0),
    val_snr_list=(0, 5, 10, 15, 20, 25), downlink_snr_db=25.0,
    ph1_w_dir=0.0, ph1_w_mse=0.0, ph2_w_dir=0.0, ph2_w_mse=0.0,
    weight_decay=1e-5, data_cache_size=1, log_interval=100, best_metric="mean_all",
)


def channel_data_api():
    """Load the accepted stdlib interface without utils/__init__ GPU side effects."""
    name = "_precoder_replayable_channel_data_v1"
    path = REPO_ROOT / "utils" / "replayable_channel_data.py"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError("Replayable data module identity collision")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load accepted replayable channel-data interface")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def make_channel_data(cfg):
    return channel_data_api().ReplayableChannelData(
        base_seed=cfg.SEED, k_users=cfg.K_USERS, nt=cfg.ANTENNAS,
        nsc=cfg.SUBCARRIERS, batch_size=cfg.BATCH_SIZE,
        validation_batch_size=cfg.VAL_BATCH, carrier_freq=cfg.CARRIER_FREQ,
        speed=cfg.SPEED, cdl_model="C", device="cuda:0",
    )


@dataclass
class ASReceiverTrainConfig(TrainConfig):
    arm: str = "full_control"
    source_identity: dict = field(default_factory=dict)
    channel_data_protocol: dict = field(default_factory=dict)

    @property
    def AS_RECEIVER_LAMBDA_FLOOR(self):
        return RECEIVER_SETTINGS["lambda_floor"]

    @property
    def AS_RECEIVER_SOLVE_DTYPE(self):
        return RECEIVER_SETTINGS["solve_dtype"]

    @property
    def architecture_metadata(self):
        receiver = self.arm == "as_receiver"
        return {
            "arm": self.arm,
            "model_class": "AdaptiveHybridPrecoderASReceiver" if receiver else "AdaptiveHybridPrecoder",
            "architecture": "as_receiver_v1" if receiver else "full",
            "ue_ffn_dim": 2048, "D_MODEL": 256, "ue_layers_per_branch": 4,
            "refinement_layers": 5, "max_dimensions": [64, 256],
            "allocation_grid": [[0, 256]], "running_pwr_shape": [1], "cfg_idx": 0,
            "receiver_settings": deepcopy(RECEIVER_SETTINGS) if receiver else None,
        }

    def to_dict(self):
        result = super().to_dict()
        result.update(
            arm=self.arm, architecture_metadata=self.architecture_metadata,
            source_identity=deepcopy(self.source_identity),
            channel_data_protocol=deepcopy(self.channel_data_protocol),
            paired_receiver_settings=deepcopy(RECEIVER_SETTINGS),
            screening_protocol={
                "screening_not_formal_result": True, "new_optimizer_updates": 20000,
                "seed": self.SEED, "planned_seeds": list(SEEDS),
                "task_path": "H_hat -> normalized RZF -> sum-rate",
                "optimizer": "AdamW", "scheduler": "OneCycleLR",
                "all_parameters_trainable": True, "shared_learning_rate": True,
                "validation_noise_seed": "seed + 1000000 + cfg_idx*10000 + FB_SNR*10",
                "fresh_post_factory_default_rng_seed": self.SEED,
                "new_receiver_dropout": 0.0,
                "resume_bitwise_training_equivalence_verified": False,
            },
        )
        return result


def as_receiver_config(arm, seed, output_root=None):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm!r}")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"Screening seed must be one of {SEEDS}")
    source = load_source_record()
    saved = source["loaded_checkpoint_config"]
    cfg = ASReceiverTrainConfig(
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
    if not isinstance(cfg, ASReceiverTrainConfig):
        raise ValueError("Expected isolated ASReceiverTrainConfig")
    expected = as_receiver_config(cfg.arm, cfg.SEED, cfg.output_root)
    if cfg.to_dict() != expected.to_dict():
        raise ValueError("Fixed ASReceiver arm/seed/recipe/source/data protocol mismatch")
    channel_data_api().validate_channel_data(cfg, make_channel_data(cfg))
