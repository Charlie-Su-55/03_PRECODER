"""Independent full-delay Joint+Swin256 screening; legacy configs unchanged."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass,field,replace
from pathlib import Path

from .train_config import BaselineDefaults,ExperimentSpec,PhysicalDefaults,ProposalModelDefaults,TrainConfig
from .as_spatial_train_config import RECIPE as _JOINT_RECIPE
from .as_receiver_train_config import channel_data_api,make_channel_data
from preflight_proposal_as_direct import load_source_record

REPO_ROOT=Path(__file__).resolve().parents[1]
OUTPUT_ROOT=REPO_ROOT/'runs_as_screen'/'proposal_joint_swin256_v1'
ARMS=('plain_swin256','joint_swin256')
SEEDS=(20261030,20261031)
RECIPE=replace(_JOINT_RECIPE,name='joint_swin256_screen20k')
SETTINGS={
    'width':128,'heads':4,'joint_layers':2,'joint_ffn_dim':256,
    'joint_ffn_activation':'GELU','normalization':'pre-LayerNorm','dropout':0,
    'complex_symbols_per_user':256,'observation_tokens':256,'joint_tokens':1024,
    'query_features':15,
    'query_feature_order':['real','imag','resource_coordinates_3','slot_coordinates_3',
                           'role_exclusive_superposed_2','user_one_hot_4','log1p_reliability_proxy'],
    'reliability_feature':'log1p(clipped_diagonal_reliability_proxy); not calibrated precision',
    'observation_features':['noisy_Y_ac','H_ul_est','resource_coordinates_3',
                            'log1p_reliability_proxy_per_user','feedback_snr_db/25'],
    'coupling':'mean_resource(abs(h_k @ h_j.conj())**2/(norm2(h_k).clamp_min(1e-12)*norm2(h_j).clamp_min(1e-12)))',
    'self_attention_bias':'same_resource*overlap_head + channel_coupling*coupling_head',
    'observation_attention_bias':'same_resource*resource_cross_head',
    'head_bias_initial_value':.02,'residual_weight_std':.001,
    'spatial_cross_output_weight_std':.001,'extra_linear_weight_std':.02,
    'residual_gate':False,'receive_power_normalization':False,
    'codeword_real_order':'all real then all imaginary','codeword_fc':[512,65536],
    'patch_size':4,'patch_grid':[64,8],'full_delay_taps':256,'antennas':32,
    'spatial_tokens_per_user':512,'spatial_cross_queries':2048,'spatial_cross_keys':1024,
    'spatial_position_features':'delay_coordinates_3+angle_coordinates_3+user_one_hot_4',
    'shared_spatial_position':True,'swin_blocks':4,'swin_heads':4,'window_size':4,
    'swin_mlp_ratio':4,'shifted_windows':False,
    'swin_implementation':'models.swin_precoder.SwinCFNetDecoder / SwinTransformerBlock',
    'inverse_transform':'models.baseline_common.inverse_angular_delay_transform',
    'attention_implementation':'torch.nn.functional.scaled_dot_product_attention',
    'query_chunk_size':64,'attention_keys':'all keys in every query chunk',
    'activation_checkpoint':True,'checkpoint_use_reentrant':False,
    'checkpoint_scope':'joint layers and per-query attention projections/chunks; spatial attention; each Swin block',
    'recovery_lambda_floor':1e-6,'recovery_solve_dtype':'complex64',
    'noise_estimator':'mean(abs(Y_ac)**2)/(1+10**(feedback_snr_db/10))',
    'noise_reduction':'all_batch_antennas_active_AS_symbols','lambda_rule':'max(noise_est,lambda_floor)',
    'physical_slice':'K*d_f:K*d_f+d_a','ue_power_ema':.02,'rzf_regularization':.001,
    'power_normalization':'per-subcarrier','old_full_bs_modules':False,
    'common_initialization':'codec and spatial_position before RNG-isolated joint extras',
    'plain_skips':['observation_memory','joint_layers','codeword_residual','spatial_cross_attention'],
}


def optimizer_group_spec(cfg=None):
    return [dict(name='bs',peak_lr=1e-4),dict(name='ue',peak_lr=2e-5)]


@dataclass
class JointSwinTrainConfig(TrainConfig):
    arm:str='plain_swin256'
    source_identity:dict=field(default_factory=dict)
    channel_data_protocol:dict=field(default_factory=dict)

    @property
    def JOINT_SWIN_LAMBDA_FLOOR(self):return 1e-6

    @property
    def JOINT_SWIN_SOLVE_DTYPE(self):return 'complex64'

    @property
    def JOINT_SWIN_QUERY_CHUNK(self):return 64

    @property
    def JOINT_SWIN_ACTIVATION_CHECKPOINT(self):return True

    @property
    def architecture_metadata(self):
        return dict(architecture='joint_swin256_v1',model_class='AdaptiveHybridPrecoderJointSwin',
            arm=self.arm,ue_frozen=False,ue_class='AdaptiveHybridEncoder',ue_ffn_dim=2048,
            ue_layers_per_branch=4,D_MODEL=256,allocation_grid=[[0,256]],max_dimensions=[64,256],
            running_pwr_shape=[1],cfg_idx=0,
            ue_power='original batch scale and EMA in train; frozen EMA in eval',
            common_bs_prefixes=['decoder.codec.','decoder.spatial_position.'],settings=deepcopy(SETTINGS),
            diagnostics='explicit return only, detached truth after H_hat and W')

    def to_dict(self):
        result=super().to_dict()
        result.update(arm=self.arm,architecture_metadata=self.architecture_metadata,
            source_identity=deepcopy(self.source_identity),channel_data_protocol=deepcopy(self.channel_data_protocol),
            optimizer_groups=optimizer_group_spec(self),screening_protocol=dict(
                screening_not_formal_result=True,new_optimizer_updates=20000,planned_seeds=list(SEEDS),
                seed=self.SEED,task_path='H_hat -> normalized RZF -> sum-rate',
                all_parameters_trainable=True,source_ue_components=['encoder','cond_embed','running_pwr'],
                source_bs_weights_loaded=False,bs_initialization='seeded random; same common BS within seed',
                automatic_mixed_precision=False,microbatching=False,full_physical_batch=48,
                checkpoint_selection='mean_all over six validation FB points, DL25',
                source_step_zero_competes=False,fresh_post_factory_default_rng_seed=self.SEED,
                resume_bitwise_training_equivalence_verified=False))
        return result


def joint_swin_config(arm,seed,output_root=None):
    if arm not in ARMS:raise ValueError(f'Unknown Joint+Swin256 arm: {arm}')
    if type(seed) is not int or seed not in SEEDS:raise ValueError(f'Seed must be one of {SEEDS}')
    source=load_source_record();saved=source['loaded_checkpoint_config']
    cfg=JointSwinTrainConfig(
        spec=ExperimentSpec(model_name='proposal',train_mode='specialist',k_users=4,
            subcarriers=256,feedback_budget=256,recipe_name=RECIPE.name,seed=seed,
            allocation=(0,256),variant=arm),allocations=((0,256),),
        full_allocation_grid=tuple(tuple(a) for a in saved['full_allocation_grid']),
        recipe=RECIPE,physical=PhysicalDefaults(**saved['physical']),
        proposal=ProposalModelDefaults(**saved['proposal_model']),
        baselines=BaselineDefaults(**saved['baseline_model']),
        output_root=OUTPUT_ROOT if output_root is None else Path(output_root).absolute(),
        arm=arm,source_identity=deepcopy(source))
    cfg.channel_data_protocol=make_channel_data(cfg).protocol
    return cfg


def validate_config(cfg):
    if not isinstance(cfg,JointSwinTrainConfig):raise ValueError('Expected isolated JointSwinTrainConfig')
    if cfg.to_dict()!=joint_swin_config(cfg.arm,cfg.SEED,cfg.output_root).to_dict():
        raise ValueError('Fixed Joint+Swin256 source/arm/recipe/architecture/data mismatch')
    channel_data_api().validate_channel_data(cfg,make_channel_data(cfg))
