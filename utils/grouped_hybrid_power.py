"""Optional transmit-energy gate; unchanged hybrid geometry and receiver mathematics.

This module is loaded by its standalone file path for standard-library dry runs.
Torch is imported only by explicit numerical helpers. No old protocol globals
are changed. The historical prior's internal statistic format is reused, while
all experiment identities and random addresses use this batch's new namespace.
"""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math
import random

from configs.grouped_hybrid_anchor_train_config import protocol as legacy

NAMESPACE='grouped_hybrid_power_v1'
ARMS=('fdma_control','h32_control','h32_static_power','h32_gamma_power')
SOURCE_PRIOR_OFFSET=20000
NODES=legacy.NODES
EMA_BETA,UNIFORM_SHRINK,PI_FLOOR,RIDGE=legacy.EMA_BETA,legacy.UNIFORM_SHRINK,legacy.PI_FLOOR,legacy.RIDGE


def source_arm(arm):
    if arm not in ARMS: raise ValueError('Unknown grouped hybrid power arm: '+str(arm))
    return 'fdma_control' if arm=='fdma_control' else 'hybrid_f32_a64'


class LayoutSpec(legacy.LayoutSpec):
    """A genuinely new arm identity with the reviewed old coordinate geometry."""
    def __post_init__(self):
        legacy.LayoutSpec(source_arm(self.arm))
    @property
    def df(self): return 64 if self.arm=='fdma_control' else 32
    @property
    def da(self): return 0 if self.arm=='fdma_control' else 64


def layout_for_arm(arm): return LayoutSpec(arm)


def namespace_seed(seed,stream):
    if type(seed) is not int or not 0<=seed<2**63 or stream not in ('channel','initialization','channel_test','channel_validation'):
        raise ValueError('Invalid power experiment seed namespace')
    body=json.dumps(dict(namespace=NAMESPACE,seed=seed,stream=stream),sort_keys=True,separators=(',',':'))
    return int.from_bytes(hashlib.sha256(body.encode()).digest()[:4],'big')


@dataclass(frozen=True)
class RandomAddress(legacy.RandomAddress):
    def to_dict(self): return dict(namespace=NAMESPACE,seed=self.seed,split=self.split,index=self.index,stream=self.stream)


def sample_gamma(seed,global_step):
    return random.Random(RandomAddress(seed,'train',global_step,'gamma').seed_value).uniform(0,25)


def standard_noise(seed,split,index,batch_size,device='cpu'):
    import torch
    if type(batch_size) is not int or batch_size<1: raise ValueError('Positive batch size required')
    generator=torch.Generator(device='cpu').manual_seed(RandomAddress(seed,split,index,'standard_noise').seed_value)
    return torch.view_as_complex(torch.randn(batch_size,32,256,2,generator=generator,dtype=torch.float32)).to(device)


noise_variance=legacy.noise_variance
interpolation=legacy.interpolation
tensor_geometry=legacy.tensor_geometry
normalize_and_map=legacy.normalize_and_map
transmit=legacy.transmit
recover=legacy.recover
make_power_prior=legacy.make_power_prior
_complex=legacy._complex


def gate_parameter_count(arm):
    source_arm(arm)
    return 1 if arm=='h32_static_power' else 6 if arm=='h32_gamma_power' else 0


def gate_value(theta,gamma_db):
    """Natural-log exclusive/shared energy multiplier, differentiable at zero."""
    import torch
    nodes=interpolation(gamma_db)
    if not torch.is_tensor(theta) or theta.ndim!=1 or theta.numel() not in (1,6) or theta.dtype!=torch.float32:
        raise ValueError('Gate requires one or six float32 scalars')
    if not torch.isfinite(theta).all(): raise FloatingPointError('Nonfinite power gate')
    value=theta[0] if theta.numel()==1 else sum(theta[node]*weight for node,weight in nodes)
    return math.log(4)*torch.tanh(value)


def gate_raw_symbols(raw,layout,g_log_energy_ratio):
    """Multiply only raw exclusive coordinates, before the old joint normalizer."""
    import torch
    _complex(raw,'raw')
    if raw.ndim!=3 or tuple(raw.shape[1:])!=(4,256): raise ValueError('raw[B,4,256] required')
    if (not torch.is_tensor(g_log_energy_ratio) or g_log_energy_ratio.ndim!=0
            or g_log_energy_ratio.dtype!=torch.float32 or g_log_energy_ratio.device!=raw.device
            or not torch.isfinite(g_log_energy_ratio)):
        raise ValueError('Finite scalar float32 gate on the raw-symbol device required')
    exclusive=tensor_geometry(layout,raw.device)['exclusive_mask']
    factor=torch.where(exclusive,torch.exp(g_log_energy_ratio/2),torch.ones((),device=raw.device))
    return raw*factor[None]


def make_power_gate(arm):
    import torch
    from torch import nn
    count=gate_parameter_count(arm)
    class SharedEnergyGate(nn.Module):
        def __init__(self):
            super().__init__()
            if count: self.theta=nn.Parameter(torch.zeros(count,dtype=torch.float32))
            else: self.register_parameter('theta',None)
        def forward(self,raw,gamma_db,layout):
            noise_variance(gamma_db)
            if self.theta is None: return raw,raw.real.new_zeros(())
            value=gate_value(self.theta,gamma_db)
            return gate_raw_symbols(raw,layout,value),value
    return SharedEnergyGate()


def prior_metadata():
    value=deepcopy(legacy.prior_metadata())
    value.update(initial='strictly inherited source best pi/count/node mass; cumulative count20000',
        source_prior_offset=SOURCE_PRIOR_OFFSET,
        timing='forward reads offset+local_step-1; successful optimizer update commits offset+local_step',
        statistic_schema_reused=legacy.NAMESPACE)
    return value


def protocol_metadata():
    value=deepcopy(legacy.protocol_metadata())
    value.update(schema=NAMESPACE,random_namespace=NAMESPACE,power_control='arm-dependent bounded shared scalar gate',
        prior=prior_metadata(),energy_gate=dict(nodes=list(NODES),constant_parameters=1,gamma_parameters=6,
        log_energy_ratio='log(4)*tanh(linearly interpolated theta)',
        exclusive_amplitude_multiplier='exp(g/2)',shared_amplitude_multiplier=1,
        relative_energy_ratio_range=[.25,4.],normalization='unchanged joint energy64 after gate',
        gate_inputs=['public gamma only'],initial_theta=0))
    return value


def architecture_metadata(arm):
    value=deepcopy(legacy.architecture_metadata(source_arm(arm)))
    value.update(architecture=NAMESPACE,model_class='GroupedHybridPowerPrecoder',arm=arm,
        layout_spec=LayoutSpec(arm).to_dict(),protocol=protocol_metadata(),prior=prior_metadata(),
        gate_parameters=gate_parameter_count(arm),source_layout=source_arm(arm),
        initialization='same-seed verified completed anchor source best, all old buffers retained; gate zeros')
    return value
