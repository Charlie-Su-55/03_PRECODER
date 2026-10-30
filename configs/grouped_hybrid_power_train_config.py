"""Independent four-arm energy-gate continuation; no historical config mutation."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import importlib.util
from pathlib import Path
import sys

from .grouped_codec_train_config import GroupedCodecTrainConfig, REPO_ROOT, SEEDS, channel_data_api
from .grouped_hybrid_anchor_train_config import anchor_config, RECIPE as SOURCE_RECIPE

ARMS = ('fdma_control', 'h32_control', 'h32_static_power', 'h32_gamma_power')
OUTPUT_ROOT = REPO_ROOT / 'runs_as_screen/grouped_hybrid_power_v1'
SOURCE_BATCH = REPO_ROOT / 'runs_as_screen/grouped_hybrid_anchor_v1/batches/batch_cf66957ea2914720bbdd9ed15dbf3884'
SOURCE_PRIOR_OFFSET = 20000
SOURCE_BEST_SHA256 = {
    ('fdma_control', 20261030): 'eb8a4514ba8fe6b2e57472c1a7c2a6bd8259855e394fb6310f8901241c42c26d',
    ('fdma_control', 20261031): '79b399b13125abc6857b7d851c9e11a80a7f2c093f061353fe878e5802de213e',
    ('hybrid_f32_a64', 20261030): 'aae67b5009a22229d0133d96ab4d10e8f8015b7fcb3319c6801ae3a439af8c76',
    ('hybrid_f32_a64', 20261031): '331a52267b376656d4c6bbf6d69e98e3bdc950137a54d68d5d7ecb39c4f150e1',
}
RECIPE = replace(SOURCE_RECIPE, name='grouped_hybrid_power_continue20k')


def feedback_api():
    name = '_grouped_hybrid_power_v1'
    path = REPO_ROOT / 'utils/grouped_hybrid_power.py'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError('Power protocol module collision')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load independent power protocol')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


protocol = feedback_api()


def source_arm(arm):
    if arm not in ARMS:
        raise ValueError('Unknown fixed power arm')
    return 'fdma_control' if arm == 'fdma_control' else 'hybrid_f32_a64'


def source_identity(arm, seed):
    cfg = anchor_config(source_arm(arm), seed)
    return dict(arm=cfg.arm, training_seed=seed, best_path=cfg.BEST_PATH,
        latest_path=cfg.SAVE_PATH, config_path=cfg.CONFIG_PATH, status_path=cfg.STATUS_PATH,
        best_sha256=SOURCE_BEST_SHA256[(cfg.arm, seed)],
        complete_source_configuration=cfg.to_dict(), completed_updates=20000,
        selected_best_step=20000, prior_update_count=SOURCE_PRIOR_OFFSET,
        batch_summary=str(SOURCE_BATCH / 'batch_summary.json'),
        selection='single six-node validation mean_all best; no test selection')


def lifecycle_descriptor():
    return dict(version='grouped-hybrid-power-lifecycle-v1', initial_validation=True,
        best_includes_optimizer_scheduler_rng=True, source_prior_offset=SOURCE_PRIOR_OFFSET,
        optimizer_step='local stage update in [0,20000]',
        prior_step='source_prior_offset + local optimizer step',
        forward_reads='prior at source_prior_offset + local step - 1',
        update='only after successful optimizer.step; actual gated jointly normalized symbols',
        updates_during_validation=False, updates_without_optimizer=False)


def execution_descriptor(seed):
    return dict(version='grouped-hybrid-power-training-v1', seed=seed,
        physical_protocol=protocol.protocol_metadata(), random_namespace='grouped_hybrid_power_v1',
        channel_base_seed=protocol.namespace_seed(seed, 'channel'),
        initialization_seed=protocol.namespace_seed(seed, 'initialization'),
        gamma='new namespace sample_gamma(seed,local_step); uniform [0,25]',
        training_noise_address=dict(split='train', index='local optimizer step'),
        validation_noise_address=dict(split='validation', index='batch_index'),
        validation_samples=256, validation_batch_size=8,
        validation_reuses_noise_across_gamma=True,
        numerical_policy=dict(real_dtype='float32', complex_dtype='complex64',
            matmul_precision='highest', allow_tf32=False, amp=False, microbatch=False))


@dataclass
class GroupedHybridPowerTrainConfig(GroupedCodecTrainConfig):
    output_root: Path = OUTPUT_ROOT
    recipe: object = RECIPE

    @property
    def layout_spec(self): return protocol.LayoutSpec(self.arm)
    @property
    def architecture_metadata(self): return protocol.architecture_metadata(self.arm)
    @property
    def EXP_NAME(self):
        return f'grouped_hybrid_power_v1_{self.arm}_K4_Nsc256_Dtot256_continue20k_seed{self.SEED}'

    def to_dict(self):
        value = super().to_dict()
        value.update(schema='grouped-hybrid-power-config-v1',
            execution_protocol=execution_descriptor(self.SEED), lifecycle_protocol=lifecycle_descriptor(),
            source_identity=source_identity(self.arm, self.SEED),
            initialization=dict(source='same-seed completed anchor best; full prior retained',
                source_prior_offset=SOURCE_PRIOR_OFFSET, new_gate_initialization='zero',
                before_optimizer=True, explicit_migration=True, resume_migration=False),
            screening_protocol=dict(new_optimizer_updates=20000, seeds=list(SEEDS),
                source_step_zero_competes=True, task_path='H_hat -> normalized RZF -> sum-rate',
                shared_ue_bs_gate_peak_lr=1e-4, validation_samples=256, validation_batch_size=8,
                selection='mean_all over six gamma points; DL25',
                full_training_bitwise_resume_verified=False))
        return value


def make_channel_data(cfg):
    return channel_data_api().ReplayableChannelData(
        base_seed=protocol.namespace_seed(cfg.SEED, 'channel'), k_users=4, nt=32, nsc=256,
        batch_size=cfg.BATCH_SIZE, validation_batch_size=cfg.VAL_BATCH,
        carrier_freq=cfg.CARRIER_FREQ, speed=cfg.SPEED, cdl_model='C', device='cuda:0')


def power_config(arm, seed, output_root=None):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unknown fixed power arm or seed')
    cfg = GroupedHybridPowerTrainConfig(arm, seed,
        OUTPUT_ROOT if output_root is None else Path(output_root).absolute())
    cfg.channel_data_protocol = make_channel_data(cfg).protocol
    return cfg


def validate_config(cfg):
    if type(cfg) is not GroupedHybridPowerTrainConfig:
        raise ValueError('Independent power config required')
    if set(vars(cfg)) != {item.name for item in fields(cfg)}:
        raise ValueError('Unserialized config overrides forbidden')
    if cfg.to_dict() != power_config(cfg.arm, cfg.SEED, cfg.output_root).to_dict():
        raise ValueError('Fixed power recipe/layout/source/protocol changed')
    if hasattr(cfg, 'ALLOCATION_GRID'):
        raise ValueError('Grouped power layouts are not legacy allocation grids')
    channel_data_api().validate_channel_data(cfg, make_channel_data(cfg))
