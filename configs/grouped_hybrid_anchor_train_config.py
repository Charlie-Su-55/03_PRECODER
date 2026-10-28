"""Fixed three-arm continuation protocol; historical configs remain unchanged."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import importlib.util
from pathlib import Path
import sys

from .grouped_codec_train_config import GroupedCodecTrainConfig, REPO_ROOT, RECIPE as OLD_RECIPE, SEEDS, channel_data_api
from .grouped_codec_fdma_train_config import fdma_config

ARMS = ('fdma_control', 'hybrid_f16_a96', 'hybrid_f32_a64')
OUTPUT_ROOT = REPO_ROOT / 'runs_as_screen/grouped_hybrid_anchor_v1'
SOURCE_BEST_SHA256 = {
    20261030: '3aab138084319dc9d875a4a8b53217eaf99eb6d6a0e42b3f13de0ff2994a010f',
    20261031: '7f3731a54fbc568378f5dbeb31800ffeb6ba25051be47a24d01f721529cd47f5',
}
RECIPE = replace(OLD_RECIPE, name='grouped_hybrid_anchor_continue20k', val_batch=256)


def feedback_api():
    name = '_grouped_hybrid_feedback_v1'
    path = REPO_ROOT / 'utils/grouped_hybrid_feedback.py'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve(): raise RuntimeError('Hybrid protocol module collision')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError('Cannot load hybrid protocol')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]; raise
    return module


protocol = feedback_api()


def lifecycle_descriptor():
    return dict(version='grouped-hybrid-anchor-lifecycle-v1', initial_validation=True,
        best_includes_optimizer_scheduler_rng=True,
        update='after successful optimizer.step, before scheduler/validation/checkpoint',
        statistic='detached jointly energy-normalized per-UE per-coordinate power',
        forward_reads='committed state at global_step-1', ema_beta=.01, uniform_shrinkage=.05,
        gamma_nodes=[0,5,10,15,20,25], interpolation='linear neighboring gamma nodes',
        updates_during_validation=False, updates_without_optimizer=False)


def execution_descriptor(seed):
    return dict(version='grouped-hybrid-anchor-training-v1', seed=seed,
        physical_protocol=protocol.protocol_metadata(),
        random_namespace='grouped_hybrid_anchor_v1',
        channel_base_seed=protocol.namespace_seed(seed,'channel'),
        initialization_seed=protocol.namespace_seed(seed,'initialization'),
        gamma='new namespace sample_gamma(seed,global_step); uniform [0,25]',
        training_noise_address=dict(split='train',index='global_step'),
        validation_noise_address=dict(split='validation',index='batch_index'),
        validation_samples=256, validation_batch_size=8,
        validation_reuses_noise_across_gamma=True,
        numerical_policy=dict(real_dtype='float32',complex_dtype='complex64',
            matmul_precision='highest',allow_tf32=False,amp=False,microbatch=False))


def source_identity(seed):
    cfg = fdma_config(seed)
    return dict(arm='legacy_fdma', training_seed=seed, best_path=cfg.BEST_PATH,
        latest_path=cfg.SAVE_PATH, config_path=cfg.CONFIG_PATH,status_path=cfg.STATUS_PATH,
        best_sha256=SOURCE_BEST_SHA256[seed], complete_source_configuration=cfg.to_dict(),
        completed_updates=20000, selection='mean_all; selected best may precede latest')


@dataclass
class GroupedHybridAnchorTrainConfig(GroupedCodecTrainConfig):
    output_root: Path = OUTPUT_ROOT
    recipe: object = RECIPE

    @property
    def layout_spec(self): return protocol.LayoutSpec(self.arm)
    @property
    def architecture_metadata(self): return protocol.architecture_metadata(self.arm)
    @property
    def EXP_NAME(self): return f'grouped_hybrid_anchor_v1_{self.arm}_K4_Nsc256_Dtot256_continue20k_seed{self.SEED}'

    def to_dict(self):
        value = super().to_dict()
        value.update(schema='grouped-hybrid-anchor-config-v1',
            execution_protocol=execution_descriptor(self.SEED), lifecycle_protocol=lifecycle_descriptor(),
            source_identity=source_identity(self.SEED),
            initialization=dict(source='same-seed completed Legacy-FDMA best',
                new_modules_seed=protocol.namespace_seed(self.SEED,'initialization'),
                before_optimizer=True, explicit_migration=True, resume_migration=False),
            screening_protocol=dict(new_optimizer_updates=20000,seeds=list(SEEDS),
                source_step_zero_competes=True,task_path='H_hat -> normalized RZF -> sum-rate',
                shared_ue_bs_peak_lr=1e-4,validation_samples=256,validation_batch_size=8,
                selection='mean_all over six gamma points; DL25',
                full_training_bitwise_resume_verified=False))
        return value


def make_channel_data(cfg):
    return channel_data_api().ReplayableChannelData(
        base_seed=protocol.namespace_seed(cfg.SEED,'channel'),k_users=4,nt=32,nsc=256,
        batch_size=cfg.BATCH_SIZE,validation_batch_size=cfg.VAL_BATCH,
        carrier_freq=cfg.CARRIER_FREQ,speed=cfg.SPEED,cdl_model='C',device='cuda:0')


def anchor_config(arm, seed, output_root=None):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS: raise ValueError('Unknown fixed hybrid arm or seed')
    cfg = GroupedHybridAnchorTrainConfig(arm,seed,OUTPUT_ROOT if output_root is None else Path(output_root).absolute())
    cfg.channel_data_protocol = make_channel_data(cfg).protocol
    return cfg


def validate_config(cfg):
    if type(cfg) is not GroupedHybridAnchorTrainConfig: raise ValueError('Independent hybrid config required')
    if set(vars(cfg)) != {item.name for item in fields(cfg)}: raise ValueError('Unserialized config overrides forbidden')
    if cfg.to_dict() != anchor_config(cfg.arm,cfg.SEED,cfg.output_root).to_dict(): raise ValueError('Fixed recipe/layout/source/protocol changed')
    if hasattr(cfg,'ALLOCATION_GRID'): raise ValueError('Grouped hybrid is not a legacy allocation grid')
    channel_data_api().validate_channel_data(cfg,make_channel_data(cfg))
