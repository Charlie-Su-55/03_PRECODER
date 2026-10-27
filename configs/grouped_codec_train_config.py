"""Independent grouped-codec layouts; no legacy Df/Da allocation proxy."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import importlib.util
from pathlib import Path
import sys

from .train_config import PhysicalDefaults, ProposalModelDefaults, TrainingRecipe
from .as_receiver_train_config import channel_data_api, make_channel_data

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / 'runs_as_screen' / 'grouped_codec_v1'
ARMS = ('legacy_all4', 'ordered_all4', 'legacy_group2', 'ordered_group2', 'ordered_fdma')
SEEDS = (20261030, 20261031)
RECIPE = TrainingRecipe(
    name='grouped_codec_random20k', total_steps=20000, batch_size=48, lr=1e-4,
    val_interval=500, val_batch=64, warmup_pct=.1, grad_clip=.5,
    phase1_pct=.4, snr_stage1_pct=.3, snr_stage2_pct=.7,
    stage1_snr=(0.,25.), stage2_snr=(0.,25.), stage3_snr=(0.,25.),
    val_snr_list=(0,5,10,15,20,25), downlink_snr_db=25.,
    ph1_w_dir=0., ph1_w_mse=0., ph2_w_dir=0., ph2_w_mse=0.,
    weight_decay=1e-5, data_cache_size=1, log_interval=100, best_metric='mean_all')


def feedback_api():
    """Real stdlib source loading without heavyweight utils/__init__."""
    name = '_grouped_feedback_protocol_v1'
    path = REPO_ROOT / 'utils/grouped_feedback_protocol.py'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError('Grouped feedback module identity collision')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load grouped feedback protocol')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


protocol = feedback_api()


def execution_descriptor(seed):
    return dict(version='grouped-codec-training-v1', seed=seed,
        physical_protocol=protocol.protocol_metadata(),
        gamma='sample_gamma(seed,global_step); continuous uniform [0,25]',
        training_noise_address=dict(split='train',index='global_step'),
        validation_noise_address=dict(split='validation',index=0),
        validation_reuses_noise_across_gamma=True, independent_of_encoder_rng=True,
        numerical_policy=dict(real_dtype='float32',complex_dtype='complex64',
            matmul_precision='highest',allow_tf32=False,amp=False,microbatch=False))


@dataclass
class GroupedCodecTrainConfig:
    """Actual independent config, never a TrainConfig or allocation disguise."""
    arm: str
    SEED: int
    output_root: Path = OUTPUT_ROOT
    recipe: TrainingRecipe = RECIPE
    physical: PhysicalDefaults = field(default_factory=PhysicalDefaults)
    proposal: ProposalModelDefaults = field(default_factory=ProposalModelDefaults)
    channel_data_protocol: dict = field(default_factory=dict)

    K_USERS = 4
    SUBCARRIERS = 256
    D_TOT = 256
    D_F_MAX_PER_USER = 64  # Original encoder storage width, NOT physical allocation.
    D_A_MAX_SHARED = 256
    TRAIN_MODE = 'independent_layout'
    RZF_ALPHA = 1e-3

    def __getattr__(self, name):
        recipe_names = {'BATCH_SIZE':'batch_size','LR':'lr','TOTAL_STEPS':'total_steps',
            'VAL_INTERVAL':'val_interval','VAL_BATCH':'val_batch','WARMUP_PCT':'warmup_pct',
            'GRAD_CLIP':'grad_clip','WEIGHT_DECAY':'weight_decay','DATA_CACHE_SIZE':'data_cache_size',
            'LOG_INTERVAL':'log_interval','BEST_METRIC':'best_metric','DIR_MODE':'dir_mode',
            'PHASE1_STEPS':'phase1_steps','STAGE1_STEPS':'stage1_steps','STAGE2_STEPS':'stage2_steps',
            'STAGE1_SNR':'stage1_snr','STAGE2_SNR':'stage2_snr','STAGE3_SNR':'stage3_snr',
            'DOWNLINK_SNR_DB':'downlink_snr_db','PH1_W_DIR':'ph1_w_dir','PH1_W_MSE':'ph1_w_mse',
            'PH2_W_DIR':'ph2_w_dir','PH2_W_MSE':'ph2_w_mse'}
        if name in recipe_names:
            return getattr(self.recipe, recipe_names[name])
        if name == 'VAL_SNR_LIST': return list(self.recipe.val_snr_list)
        for field_name in ('physical','proposal'):
            owner = object.__getattribute__(self, field_name)
            if hasattr(owner, name): return getattr(owner, name)
        raise AttributeError(name)

    @property
    def layout_spec(self): return protocol.layout_for_arm(self.arm)
    @property
    def architecture_metadata(self): return protocol.architecture_metadata(self.arm)
    @property
    def EXP_NAME(self): return f'grouped_codec_{self.arm}_K4_Nsc256_Dtot256_random20k_seed{self.SEED}'
    @property
    def SAVE_DIR(self): return str(self.output_root/'checkpoints'/self.EXP_NAME)
    @property
    def LOG_DIR(self): return str(self.output_root/'logs'/self.EXP_NAME)
    @property
    def SAVE_PATH(self): return str(Path(self.SAVE_DIR)/'latest.pth')
    @property
    def BEST_PATH(self): return str(Path(self.SAVE_DIR)/'best.pth')
    @property
    def CONFIG_PATH(self): return str(Path(self.SAVE_DIR)/'config.json')
    @property
    def STATUS_PATH(self): return str(Path(self.SAVE_DIR)/'status.json')
    @property
    def METRICS_PATH(self): return str(Path(self.LOG_DIR)/'metrics.jsonl')

    def to_dict(self):
        return dict(schema='grouped-codec-config-v1',arm=self.arm,seed=self.SEED,
            scenario=dict(k_users=4,nt=32,nsc=256,total_occupied_re=256,nsb=4,cdl_model='C'),
            physical=asdict(self.physical),proposal_model=asdict(self.proposal),recipe=asdict(self.recipe),
            output_root=str(self.output_root),experiment=self.EXP_NAME,
            layout_spec=self.layout_spec.to_dict(),execution_protocol=execution_descriptor(self.SEED),
            channel_data_protocol=deepcopy(self.channel_data_protocol),architecture_metadata=self.architecture_metadata,
            initialization=dict(source='random',historical_checkpoint=None,independent_bs_stream=True,
                same_ue_within_family=True,post_factory_default_rng_seed=self.SEED),
            screening_protocol=dict(new_optimizer_updates=20000,seeds=list(SEEDS),source_step_zero_competes=False,
                task_path='H_hat -> normalized RZF -> sum-rate',shared_ue_bs_peak_lr=1e-4,
                automatic_mixed_precision=False,microbatching=False,precision='float32 / complex64',
                matmul_precision='highest',allow_tf32=False,validation_axis='gamma_db',
                selection='mean_all over six gamma points; DL25',full_training_bitwise_resume_verified=False))


def grouped_codec_config(arm,seed,output_root=None):
    if arm not in ARMS: raise ValueError(f'Unknown grouped codec arm: {arm}')
    if type(seed) is not int or seed not in SEEDS: raise ValueError(f'Seed must be one of {SEEDS}')
    cfg=GroupedCodecTrainConfig(arm,seed,OUTPUT_ROOT if output_root is None else Path(output_root).absolute())
    cfg.channel_data_protocol=make_channel_data(cfg).protocol
    return cfg


def validate_config(cfg):
    if not isinstance(cfg,GroupedCodecTrainConfig): raise ValueError('Expected independent grouped codec config')
    if cfg.to_dict()!=grouped_codec_config(cfg.arm,cfg.SEED,cfg.output_root).to_dict():
        raise ValueError('Fixed grouped codec arm/seed/layout/recipe/protocol mismatch')
    if hasattr(cfg,'ALLOCATION_GRID'): raise ValueError('Grouped layouts cannot use a legacy allocation grid')
    channel_data_api().validate_channel_data(cfg,make_channel_data(cfg))
