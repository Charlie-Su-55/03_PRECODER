"""F: Legacy AS encoder + FDMA support; reuse the grouped numerical path."""
from __future__ import annotations

import torch
from torch import nn

from configs.grouped_codec_fdma_train_config import protocol, validate_config
from .grouped_codec_precoder import (
    GroupedCodecPrecoder, LegacyUE, GroupedJointSwinDecoder,
)


class GroupedCodecFDMAPrecoder(GroupedCodecPrecoder):
    """Only construction/identity differ; all physical and decoder math is inherited."""

    def __init__(self, cfg):
        # Calling the old constructor would reject F; do not disguise F as A/E.
        nn.Module.__init__(self)
        validate_config(cfg)
        self.cfg = cfg
        self.arm = cfg.arm
        self.layout = cfg.layout_spec
        self.ue_frozen = False
        # Exactly the existing Legacy-family and common-BS init addresses.
        # F's arm name is deliberately absent from both random addresses.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                protocol.RandomAddress(cfg.SEED, 'init', 0, 'ue_legacy').seed_value)
            self.ue = LegacyUE(cfg)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                protocol.RandomAddress(cfg.SEED, 'init', 0, 'bs').seed_value)
            self.decoder = GroupedJointSwinDecoder(self.layout)

    @property
    def architecture_metadata(self):
        return self.cfg.architecture_metadata
