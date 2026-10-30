"""Isolated bounded exclusive/shared energy gate on the complete anchor model."""
from __future__ import annotations

import math
import torch
from torch import nn

from configs.grouped_hybrid_power_train_config import protocol,validate_config
from .grouped_hybrid_anchor import GroupedHybridAnchorPrecoder,HybridJointSwinDecoder
from .grouped_codec_precoder import LegacyUE
from .baseline_common import inverse_angular_delay_transform


class GroupedHybridPowerPrecoder(GroupedHybridAnchorPrecoder):
    """Only theta is new; complete UE/decoder/prior state names are unchanged.

Training statistics use *cumulative* update indices. The independent trainer's
execution/lifecycle adapter translates local optimizer step to source offset
plus local step exactly once; evaluation never commits these statistics.
    """
    def __init__(self,cfg):
        nn.Module.__init__(self)
        validate_config(cfg)
        self.cfg,self.arm,self.layout=cfg,cfg.arm,protocol.LayoutSpec(cfg.arm)
        self.ue_frozen=False
        seed=protocol.namespace_seed(cfg.SEED,'initialization')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(protocol.RandomAddress(seed,'init',0,'ue_legacy').seed_value)
            self.ue=LegacyUE(cfg)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(protocol.RandomAddress(seed,'init',0,'bs').seed_value)
            self.decoder=HybridJointSwinDecoder(self.layout,cfg.SEED)
        self.power_prior=protocol.make_power_prior(self.layout)
        # torch.zeros consumes no random numbers, leaving common initialization paired.
        self.power_gate=protocol.make_power_gate(cfg.arm)

    @property
    def architecture_metadata(self): return protocol.architecture_metadata(self.arm)

    def ue_parameters(self):
        yield from self.ue.parameters()
        yield from self.power_gate.parameters()

    def gate_log_ratio(self,gamma_db):
        protocol.noise_variance(gamma_db)
        if self.power_gate.theta is None: return self.power_prior.pi_nodes.new_zeros(())
        return protocol.gate_value(self.power_gate.theta,gamma_db)

    def _forward(self,H_dl,H_ul,gamma_db,*,standard_noise,H_ul_est=None,diagnostics=False,statistics_step=None):
        batch=H_dl.shape[0]
        protocol._complex(H_dl,'DL',(batch,4,32,256))
        protocol._complex(H_ul,'UL',(batch,4,32,256))
        if H_dl.device!=H_ul.device or batch<1: raise ValueError('Invalid paired channels')
        if H_ul_est is not None and not torch.equal(H_ul_est,H_ul): raise ValueError('Perfect UL required')
        N0=protocol.noise_variance(gamma_db)
        prior=self.power_prior.read(gamma_db)
        raw=self.ue(H_dl,gamma_db,self.layout.active_per_user)
        gated,g_value=self.power_gate(raw,gamma_db,self.layout)
        X,truth,energy=protocol.normalize_and_map(gated,self.layout)
        Y=protocol.transmit(H_ul,X,self.layout,N0=N0,standard_noise=standard_noise)
        recovered=protocol.recover(Y,H_ul,self.layout,N0=N0,prior_power=prior)
        H_ad,corrected,execution,anchor=self.decoder(recovered['z_hat'],recovered['error_feature'],Y,H_ul,gamma_db)
        H_hat=inverse_angular_delay_transform(H_ad,256)
        W=self.zf_solver(H_hat)
        W=W*torch.sqrt(1/(W.abs().square().sum((1,2),keepdim=True)+1e-9))
        pending=None if statistics_step is None else self.power_prior.pending(truth,gamma_db,statistics_step)
        detail=None
        if diagnostics:
            exclusive=self.decoder.exclusive_mask;shared=self.decoder.shared_mask
            power=truth.detach().abs().square()
            energy_exclusive=power.masked_fill(~exclusive[None],0).sum(-1)
            energy_shared=power.masked_fill(~shared[None],0).sum(-1)
            theta=self.power_gate.theta
            detail=dict(z_hat=recovered['z_hat'].detach(),z_corrected=corrected.detach(),true_z=truth.detach(),
                error_proxy=recovered['error_proxy'].detach(),prior_power=prior.detach(),
                error_feature=recovered['error_feature'].detach(),active_mask=self.decoder.active_mask.detach(),
                exclusive_mask=exclusive.detach(),shared_mask=shared.detach(),energy=energy.detach(),
                energy_exclusive=energy_exclusive,energy_shared=energy_shared,
                exclusive_energy_per_active_symbol=energy_exclusive/self.layout.df,
                shared_energy_per_active_symbol=None if not self.layout.da else energy_shared/self.layout.da,
                g_log_energy_ratio=g_value.detach(),
                power_gate_theta=energy.new_empty(0) if theta is None else theta.detach(),
                physical_send=X.detach(),physical_received=Y.detach(),
                anchor_rms=anchor.detach().square().mean(-1).sqrt(),
                noise_power=(standard_noise*math.sqrt(N0/2)).abs().square().mean((1,2)).detach(),
                N0=N0,gamma_db=float(gamma_db),prior_update_count=int(self.power_prior.update_count.item()),execution=execution)
        return W,H_hat,pending,detail


def migrate_source_state_dict(model,source_state):
    """Exact old best -> new stage migration; never used for new-stage resume."""
    if not isinstance(model,GroupedHybridPowerPrecoder): raise TypeError('Power target model required')
    expected=model.state_dict()
    gate_keys={'power_gate.theta'} if model.power_gate.theta is not None else set()
    common=set(expected)-gate_keys
    if not isinstance(source_state,dict) or set(source_state)!=common:
        actual=set(source_state) if isinstance(source_state,dict) else set()
        raise ValueError(f'Exact complete anchor source required; missing={sorted(common-actual)}, extra={sorted(actual-common)}')
    assembled={}
    for name,reference in expected.items():
        if name in gate_keys:
            assembled[name]=torch.zeros_like(reference)
            continue
        value=source_state[name]
        if not torch.is_tensor(value) or value.shape!=reference.shape or value.dtype!=reference.dtype:
            raise ValueError('Source shape/dtype mismatch: '+name)
        if not torch.isfinite(value).all(): raise ValueError('Nonfinite source tensor: '+name)
        assembled[name]=value
    if int(assembled['power_prior.update_count'].item())!=protocol.SOURCE_PRIOR_OFFSET:
        raise ValueError('Completed source cumulative prior count must be20000')
    model.load_state_dict(assembled,strict=True)
    model.power_prior.validate_state(expected_step=protocol.SOURCE_PRIOR_OFFSET)
    current=model.state_dict()
    for name in common:
        if not torch.equal(current[name].detach().cpu(),source_state[name].detach().cpu()):
            raise RuntimeError('Migration altered source tensor: '+name)
    if any(torch.count_nonzero(current[name]).item() for name in gate_keys):
        raise RuntimeError('Migration did not initialize exact zero gate')
    return dict(strict_loaded=True,copied_keys=sorted(common),new_parameter_keys=sorted(gate_keys),
        new_parameter_count=protocol.gate_parameter_count(model.arm),source_prior_count=protocol.SOURCE_PRIOR_OFFSET,
        prior_reset=False,new_optimizer_required=True,new_scheduler_required=True)
