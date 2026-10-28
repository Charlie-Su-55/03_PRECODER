"""Isolated F: the existing Legacy AS encoder on the true FDMA RE layout."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from .grouped_codec_train_config import (
    GroupedCodecTrainConfig, REPO_ROOT, RECIPE, SEEDS, channel_data_api,
    execution_descriptor, make_channel_data, protocol,
)

ARMS = ('legacy_fdma',)
ARCHITECTURE = 'grouped_codec_fdma_v1'
OUTPUT_ROOT = REPO_ROOT / 'runs_as_screen' / ARCHITECTURE


def fdma_architecture_metadata():
    """Reuse the fixed Legacy/BS definitions, not an A/E experiment identity."""
    metadata = protocol.architecture_metadata('legacy_all4')
    metadata.update(architecture=ARCHITECTURE, model_class='GroupedCodecFDMAPrecoder',
                    arm=ARMS[0], layout_spec=protocol.LayoutSpec('fdma').to_dict())
    return metadata


@dataclass
class GroupedCodecFDMATrainConfig(GroupedCodecTrainConfig):
    output_root: Path = OUTPUT_ROOT

    @property
    def layout_spec(self): return protocol.LayoutSpec('fdma')

    @property
    def architecture_metadata(self): return fdma_architecture_metadata()

    @property
    def EXP_NAME(self):
        return f'grouped_codec_fdma_v1_{self.arm}_K4_Nsc256_Dtot256_random20k_seed{self.SEED}'

    def to_dict(self):
        result = super().to_dict()
        result['schema'] = 'grouped-codec-fdma-config-v1'
        result['extension'] = dict(arm='legacy_fdma',
            encoder='Legacy AS branch; stored original FDMA branch remains inactive',
            reused_physical_protocol=protocol.PROTOCOL,
            independent_output_root=True, previous_arms_unchanged=True)
        return result


def fdma_config(seed, output_root=None):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f'Seed must be one of {SEEDS}')
    root = OUTPUT_ROOT if output_root is None else Path(output_root).absolute()
    cfg = GroupedCodecFDMATrainConfig(ARMS[0], seed, root)
    cfg.channel_data_protocol = make_channel_data(cfg).protocol
    return cfg


def validate_config(cfg):
    if type(cfg) is not GroupedCodecFDMATrainConfig:
        raise ValueError('Expected the independent F Legacy+FDMA configuration')
    if set(vars(cfg)) != {item.name for item in fields(cfg)}:
        raise ValueError('Unserialized runtime overrides are not part of the fixed F configuration')
    if cfg.to_dict() != fdma_config(cfg.SEED, cfg.output_root).to_dict():
        raise ValueError('Fixed F arm/seed/layout/recipe/protocol mismatch')
    if hasattr(cfg, 'ALLOCATION_GRID'):
        raise ValueError('F uses a physical LayoutSpec, not a legacy allocation proxy')
    channel_data_api().validate_channel_data(cfg, make_channel_data(cfg))
