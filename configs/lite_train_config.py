"""Standard-library-only configuration for the first isolated Lite experiment.

The existing recipe, suites, TrainConfig snapshots and their fingerprints are
unchanged. Importing this module or constructing a config creates no directory
and imports neither a model nor a numerical runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from .train_config import (
    ExperimentSpec,
    TRAINING_RECIPES,
    TrainConfig,
    resolve_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LITE_ROOT = REPO_ROOT / "runs_as_screen" / "proposal_lite"
RECIPE = "k4_n256_standard"


@dataclass
class LiteTrainConfig(TrainConfig):
    """Resolved base config plus fingerprinted, Lite-specific architecture.

    These fields intentionally remain dataclass fields: a changed architecture
    produces a changed snapshot/fingerprint, and can be rejected before any
    checkpoint loading. They are not added to the original TrainConfig.
    """

    model_class: str = "AdaptiveHybridPrecoderLite"
    architecture: str = "ueffn1024"
    ue_ffn_dim: int = 1024

    @property
    def architecture_metadata(self) -> dict:
        return {
            "model_class": self.model_class,
            "architecture": self.architecture,
            "ue_ffn_dim": self.ue_ffn_dim,
            "ue_layers": {
                "fdma": self.NUM_ENCODER_LAYERS,
                "as": self.NUM_ENCODER_LAYERS,
            },
            "D_MODEL": self.D_MODEL,
        }

    def to_dict(self) -> dict:
        snapshot = super().to_dict()
        snapshot["architecture_metadata"] = self.architecture_metadata
        return snapshot


def lite_config(output_root: str | Path | None = None) -> LiteTrainConfig:
    """Resolve the fixed K4, pure-AS Lite1024 specialist without filesystem writes.

    output_root is useful for isolated tests. The training CLI separately
    enforces resolved-path and symlink containment under the dedicated
    proposal_lite root before allowing any writes.
    """
    root = LITE_ROOT if output_root is None else Path(output_root).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()
    base = resolve_experiment(
        ExperimentSpec(
            model_name="proposal",
            train_mode="specialist",
            k_users=4,
            subcarriers=256,
            feedback_budget=256,
            recipe_name=RECIPE,
            seed=42,
            allocation=(0, 256),
            variant="ueffn1024",
        ),
        output_root=root,
    )
    # Preserve the exact resolved objects, especially the original recipe.
    cfg = LiteTrainConfig(**{
        item.name: getattr(base, item.name) for item in fields(TrainConfig)
    })
    validate_lite_recipe(cfg)
    return cfg


def validate_lite_recipe(cfg: LiteTrainConfig) -> None:
    """Reject changes outside the one prepared experiment before training.

    This validates the formal singleton training config, not a separate CUDA
    diagnostic config with a full allocation grid. No config is mutated.
    """
    if not isinstance(cfg, LiteTrainConfig):
        raise ValueError("The Lite entry requires a LiteTrainConfig.")
    if type(cfg.ue_ffn_dim) is not int:
        raise ValueError('Lite ue_ffn_dim must be the integer 1024.')
    if cfg.recipe is not TRAINING_RECIPES[RECIPE]:
        raise ValueError(f"Lite must reuse the original {RECIPE} recipe object.")

    expected_spec = ExperimentSpec(
        model_name="proposal", train_mode="specialist", k_users=4,
        subcarriers=256, feedback_budget=256, recipe_name=RECIPE,
        seed=42, allocation=(0, 256), variant="ueffn1024",
    )
    problems = []
    if cfg.spec != expected_spec:
        problems.append("expected the fixed K4 pure-AS specialist spec with seed42/ueffn1024")
    if cfg.allocations != ((0, 256),) or cfg.GAMMA_GRID != [(0, 256)]:
        problems.append("training allocations must be the singleton ((0, 256),), with index 0")
    expected_full_grid = ((64, 0), (48, 64), (32, 128), (16, 192), (0, 256))
    if cfg.full_allocation_grid != expected_full_grid:
        problems.append("the full five-point capacity grid must be retained")

    expected = {
        "model_class": "AdaptiveHybridPrecoderLite",
        "architecture": "ueffn1024",
        "ue_ffn_dim": 1024,
        "K_USERS": 4,
        "ANTENNAS": 32,
        "SUBCARRIERS": 256,
        "D_TOT": 256,
        "NUM_SUBBANDS": 4,
        "D_F_MAX_PER_USER": 64,
        "D_A_MAX_SHARED": 256,
        "D_MODEL": 256,
        "NUM_ENCODER_LAYERS": 4,
        "NUM_HEADS": 8,
        "NUM_UNFOLD_LAYERS": 5,
        "GNN_AGG_DIM": 48,
        "DROPOUT": 0.1,
        "TOTAL_POWER": 1.0,
        "TOTAL_STEPS": 50_000,
        "BATCH_SIZE": 48,
        "LR": 1e-4,
        "WEIGHT_DECAY": 1e-4,
        "WARMUP_PCT": 0.35,
        "GRAD_CLIP": 1.0,
        "VAL_INTERVAL": 500,
        "VAL_BATCH": 32,
        "VAL_SNR_LIST": [0, 10, 20, 25],
        "BEST_METRIC": "mean_all",
        "PHASE1_STEPS": 20_000,
        "PH1_W_DIR": 50.0,
        "PH1_W_MSE": 100.0,
        "PH2_W_DIR": 5.0,
        "PH2_W_MSE": 0.0,
        "STAGE1_STEPS": 15_000,
        "STAGE2_STEPS": 35_000,
        "STAGE1_SNR": (20.0, 25.0),
        "STAGE2_SNR": (10.0, 25.0),
        "STAGE3_SNR": (0.0, 25.0),
        "DOWNLINK_SNR_DB": None,
        "DIR_MODE": "real",
    }
    for name, value in expected.items():
        actual = getattr(cfg, name)
        if actual != value:
            problems.append(f"{name} must be {value!r}, got {actual!r}")
    if problems:
        raise ValueError("Invalid Lite training config: " + "; ".join(problems))
