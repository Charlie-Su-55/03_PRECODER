"""A detached physical symbol head beside the unchanged Base2 semantic path.

Only the two small MLPs are new.  The explicit Base2 forward glue below reuses
its validation, observation/query helpers, original Joint weights, shared FC,
spatial attention and Swin.  It exposes optional stages through the existing
third return value; it never stores a forward cache or accepts simulator truth.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from models.twc_v1_k8_decoder_refinement import (
    K8DecoderRefinementReceiver,build_ordered_codec,residual_ops,
    _run,_channel_coupling,K_USERS,N_ANTENNAS,N_SUBCARRIERS,MAX_SYMBOLS_PER_USER,QUERY_CHUNK,
)

ARCHITECTURE='twc_v1_k8_physical_split_v1'
VARIANTS=('control','split')
PHYSICAL_HIDDEN=64
EXPECTED_HEAD_PARAMETERS=8386
EXPECTED_BRIDGE_PARAMETERS=514


def make_physical_modules(width=128,hidden=PHYSICAL_HIDDEN):
    """Normally initialized hidden layers, with only the final layers zero."""
    if type(width) is not int or width<1 or hidden!=64:
        raise ValueError('Positive context width and the fixed64-unit MLP required')
    head=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,2))
    bridge=nn.Sequential(nn.Linear(5,hidden),nn.GELU(),nn.Linear(hidden,2))
    for module in (head[-1],bridge[-1]):
        nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
    return head,bridge


def new_state_keys(*,prefix='joint_decoder.'):
    return tuple(prefix+module+'.'+layer+'.'+field for module in ('physical_head','physical_bridge')
                 for layer in ('0','2') for field in ('weight','bias'))


def physical_split_values(head,bridge,soft,context,z_sem,reliability,active,residual_builder,*,
                          bridge_enabled,physical_from_soft=False):
    """Production gradient boundary, also directly testable with tiny tensors.

    Physical loss can reach only head parameters.  CSI loss sees the semantic
    path and bridge; every physical bridge input is detached.  Residual solves
    are metric/features only, so they do not retain an unnecessary solve graph.
    """
    if type(bridge_enabled) is not bool or type(physical_from_soft) is not bool:
        raise ValueError('Explicit boolean physical-split options required')
    if (soft.dtype!=torch.complex64 or z_sem.dtype!=soft.dtype or soft.ndim!=3
            or z_sem.shape!=soft.shape or context.shape[:-1]!=soft.shape
            or reliability.shape!=soft.shape or active.dtype!=torch.bool
            or active.shape not in (soft.shape,soft.shape[1:])):
        raise ValueError('Physical/semantic packed token axes must agree')
    if not all(bool(torch.isfinite(value).all()) for value in (soft,context,z_sem,reliability)):
        raise ValueError('Physical-split inputs must be finite')
    mask=active.to(soft.device)
    if mask.ndim==2:mask=mask[None].expand(soft.shape[0],-1,-1)
    baseline=soft.detach().masked_fill(~mask,0)
    correction=head(context.detach())
    estimate=(baseline+torch.complex(correction[...,0],correction[...,1])).masked_fill(~mask,0)
    if physical_from_soft:estimate=baseline
    with torch.no_grad():
        received=residual_builder(estimate.detach())
        direction=received['correction'].detach();energy=received['residual_energy'].detach()
    if (direction.shape!=soft.shape or direction.dtype!=soft.dtype or energy.shape!=(soft.shape[0],)
            or not bool(torch.isfinite(direction).all() & torch.isfinite(energy).all())):
        raise ValueError('Physical residual feature axes or values are invalid')
    features=torch.cat((torch.view_as_real((estimate-baseline).contiguous()),
        torch.view_as_real(direction.contiguous()),reliability[...,None]),dim=-1)
    features=features.masked_fill(~mask[...,None],0).detach()
    if bridge_enabled:
        semantic_correction=bridge(features)
        reconstructed=(z_sem+torch.complex(semantic_correction[...,0],semantic_correction[...,1])).masked_fill(~mask,0)
    else:
        reconstructed=z_sem.masked_fill(~mask,0)
    return dict(soft=baseline,x_phy=estimate,z_recon=reconstructed,d_phy=direction,
                residual_energy=energy,bridge_enabled=bridge_enabled,physical_from_soft=physical_from_soft)


def _packed_truth(output):
    resource,kind,_,active,layout_kind,_=residual_ops.packed_layout_map(output['layout'])
    tx=output['tx']
    if layout_kind=='fdma':
        data=tx['data_symbols_tx'].detach()
        if data.dtype!=torch.complex64 or data.ndim!=3 or tuple(data.shape[1:])!=(8,24):
            raise ValueError('Actual normalized FDMA symbols must be complex64[B,8,24]')
        truth=torch.zeros(data.shape[0],8,164,dtype=data.dtype,device=data.device)
        truth[...,:24]=data
    else:
        key=tx['key_symbols_tx'].detach();shared=tx['shared_symbols_tx'].detach()
        if (key.dtype!=torch.complex64 or shared.dtype!=key.dtype or key.ndim!=3
                or tuple(key.shape[1:])!=(8,4) or tuple(shared.shape)!=(key.shape[0],8,output['layout'].shared_dimension)):
            raise ValueError('Actual normalized Hybrid key/shared symbol axes required')
        truth=torch.zeros(key.shape[0],8,164,dtype=key.dtype,device=key.device)
        truth[...,:4]=key;truth[...,4:4+shared.shape[-1]]=shared
    if not bool(torch.isfinite(truth).all()):raise ValueError('Actual transmitted symbol labels must be finite')
    return truth,resource,kind.to(truth.device),active.to(truth.device),layout_kind


def _branch_errors(prediction,truth,kind,active):
    if prediction.shape!=truth.shape or prediction.dtype!=truth.dtype or not bool(torch.isfinite(prediction).all()):
        raise ValueError('Packed symbol/latent diagnostic axes or values are invalid')
    result={}
    for label,mask in [('all',active)]+[(name,active&(kind==value)) for name,value in (('key',0),('shared',1),('fdma',2))]:
        if not bool(mask.any()):continue
        error=(prediction-truth).masked_fill(~mask[None],0).abs().square().sum((1,2))
        denominator=truth.masked_fill(~mask[None],0).abs().square().sum((1,2))
        if not bool(torch.isfinite(denominator).all() & (denominator>0).all()):
            raise ValueError('A supervised branch requires positive finite actual transmitted energy')
        result[label]=error/denominator
    return result


def physical_codeword_loss(output):
    """Linear branch-normalized error per complete MU sample, then batch mean.

    FDMA uses its24 real data slots. Hybrid averages key/shared branch NMSE
    equally, rather than weighting by dimension or branch energy. Labels detach.
    """
    truth,_,kind,active,layout_kind=_packed_truth(output)
    stages=output['receiver_execution']['stages']
    if not torch.equal(stages['active_mask'].to(active.device),active):
        raise ValueError('Supervision mask differs from actual physical layout')
    errors=_branch_errors(stages['x_phy'],truth,kind,active)
    per_mu=errors['fdma'] if layout_kind=='fdma' else .5*(errors['key']+errors['shared'])
    return per_mu.mean()


def physical_diagnostics(output):
    """Output-only truth diagnostics: physical NMSE and semantic latent distance.

    All distances are linear ratios per MU sample. Received residuals are mean
    power over data REs/antennas, without received-power normalization. Latent
    distances must never be described as physical symbol reconstruction error.
    """
    with torch.no_grad():
        truth,resource,kind,active,layout_kind=_packed_truth(output)
        stages=output.get('receiver_execution',{}).get('stages',{})
        initial=torch.zeros_like(truth);soft=output['soft_symbols']
        if layout_kind=='fdma':initial[...,:24]=soft['data_symbols_hat']
        else:
            initial[...,:4]=soft['key_symbols_hat']
            initial[...,4:4+output['layout'].shared_dimension]=soft['shared_symbols_hat']
        initial=stages.get('soft',initial)
        z_sem=stages.get('z_sem',stages.get('x1',output['corrected_symbols']))
        z_recon=stages.get('z_recon',stages.get('x2',output['corrected_symbols']))
        result={}
        groups=[('physical_soft_codeword_nmse_linear',initial),
                ('latent_z_sem_distance_linear',z_sem),('latent_z_recon_distance_linear',z_recon)]
        if 'x_phy' in stages:groups.append(('physical_head_codeword_nmse_linear',stages['x_phy']))
        for prefix,prediction in groups:
            for label,value in _branch_errors(prediction,truth,kind,active).items():result[prefix+'_'+label]=value.detach()
        result['rx_residual_soft_mean_power_unnormalized']=residual_ops.data_received_residual(initial,
            output['Y_rx'],output['H_ul_est'],resource,active,pilot_positions=output['layout'].pilot_positions)['energy']
        if 'x_phy' in stages:
            name='rx_residual_physical_mean_power_unnormalized'
            result[name]=stages[name].detach() if name in stages else residual_ops.data_received_residual(
                stages['x_phy'].detach(),output['Y_rx'],output['H_ul_est'],resource,active,
                pilot_positions=output['layout'].pilot_positions)['energy']
        if any(not bool(torch.isfinite(value).all()) for value in result.values()):
            raise FloatingPointError('Nonfinite physical/latent offline diagnostic')
        return result


def migrate_source_state(model,source_state):
    """Strict complete Base2 migration, permitting only the eight new tensors."""
    new=set(new_state_keys());template=model.state_dict()
    if not new<=set(template) or not isinstance(source_state,dict) or set(source_state)!=set(template)-new:
        raise ValueError('Physical-split source must contain exactly the complete old Base2 state')
    converted={};aliases={}
    for name,expected in template.items():
        value=expected if name in new else source_state[name]
        if (not torch.is_tensor(value) or value.shape!=expected.shape or value.dtype!=expected.dtype
                or not bool(torch.isfinite(value).all())):raise ValueError('Source tensor shape/dtype/finite mismatch: '+name)
        storage=(expected.device,expected.untyped_storage().data_ptr(),expected.storage_offset(),tuple(expected.shape),tuple(expected.stride()),expected.dtype)
        if name not in new and storage in aliases and not torch.equal(value,source_state[aliases[storage]]):
            raise ValueError('Conflicting shared decoder aliases: '+aliases[storage]+' and '+name)
        if name not in new:aliases.setdefault(storage,name)
        converted[name]=value.detach().clone()
    model.load_state_dict(converted,strict=True)
    for name,value in source_state.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(),value.detach().cpu()):
            raise RuntimeError('Migration changed an old source tensor: '+name)
    return dict(strict=True,source_values_preserved=True,added_keys=sorted(new),
                new_parameters=sum(p.numel() for name,p in model.named_parameters() if name in new))


def build_codec(variant='split',*,new_module_seed=20261107,include_stages=True,**kwargs):
    if variant not in VARIANTS:raise ValueError('An explicit control/split variant is required')
    codec=build_ordered_codec(ordered=True,**kwargs)
    codec.joint_decoder=K8PhysicalSplitReceiver(codec.decoder,variant=variant,
        new_module_seed=new_module_seed,include_stages=include_stages)
    if codec.decoder is not codec.joint_decoder.base_decoder:raise RuntimeError('Shared spatial decoder alias lost')
    return codec


class K8PhysicalSplitReceiver(K8DecoderRefinementReceiver):
    def __init__(self,base_decoder,*,variant='split',new_module_seed=20261107,include_stages=True,
                 query_chunk=QUERY_CHUNK,activation_checkpoint=True):
        if variant not in VARIANTS:raise ValueError('An explicit control/split variant is required')
        super().__init__(base_decoder,variant='base2',new_module_seed=new_module_seed,
            include_stages=include_stages,query_chunk=query_chunk,activation_checkpoint=activation_checkpoint)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(new_module_seed)
            self.physical_head,self.physical_bridge=make_physical_modules(self.width)
        self.variant=variant;self.bridge_override=None;self.physical_from_soft=False
        self.physical_bridge.requires_grad_(variant=='split')

    @contextmanager
    def diagnostic_options(self,*,bridge_enabled=None,physical_from_soft=None):
        for value in (bridge_enabled,physical_from_soft):
            if value is not None and type(value) is not bool:raise ValueError('Diagnostic options must be booleans or None')
        if self.variant=='control' and bridge_enabled is True:raise ValueError('Control bridge is permanently disabled')
        old=(self.bridge_override,self.physical_from_soft)
        if bridge_enabled is not None:self.bridge_override=bridge_enabled
        if physical_from_soft is not None:self.physical_from_soft=physical_from_soft
        try:yield self
        finally:self.bridge_override,self.physical_from_soft=old

    def _physical_interface(self,soft,context,z_sem,reliability,mask,residual_builder,*,
                            bridge_enabled=None,physical_from_soft=None):
        enabled=self.variant=='split' if self.bridge_override is None else self.bridge_override
        if bridge_enabled is not None:enabled=bridge_enabled
        from_soft=self.physical_from_soft if physical_from_soft is None else physical_from_soft
        if self.variant=='control' and enabled:raise ValueError('Control bridge is permanently disabled')
        return physical_split_values(self.physical_head,self.physical_bridge,soft,context,z_sem,
            reliability,mask,residual_builder,bridge_enabled=enabled,physical_from_soft=from_soft)

    def forward(self, key, shared, *, group_count, Y_rx, H_ul_est, layout,
                feedback_gamma_db, noise_variance, pilot_energy_per_ue,
                key_energy_per_ue, shared_energy_per_ue, channel_error_variance,
                key_reliability, shared_reliability, return_stages=None,
                bridge_enabled=None, physical_from_soft=None):
        resource, kind, group, active, layout_kind, mapped_group_count = self._slot_map(layout)
        if group_count != mapped_group_count:
            raise ValueError("group_count and layout disagree")
        batch = key.shape[0] if torch.is_tensor(key) and key.ndim else 0
        if key.dtype != torch.complex64 or tuple(key.shape) != (batch, K_USERS, 4):
            raise ValueError("key must be complex64 [B,8,4]")
        expected_shared = 160 if layout_kind == "fdma" else int(active[0, 4:].sum())
        if shared.dtype != torch.complex64 or tuple(shared.shape) != (batch, K_USERS, expected_shared):
            raise ValueError(f"shared must be complex64 [B,8,{expected_shared}]")
        if Y_rx.dtype != torch.complex64 or tuple(Y_rx.shape) != (batch, N_ANTENNAS, N_SUBCARRIERS):
            raise ValueError("Y_rx must be complex64 [B,32,256]")
        if H_ul_est.dtype != torch.complex64 or tuple(H_ul_est.shape) != (batch, K_USERS, N_ANTENNAS, N_SUBCARRIERS):
            raise ValueError("H_ul_est must be complex64 [B,8,32,256]")
        device = self.base_decoder.fc.weight.device
        if any(t.device != device for t in (key, shared, Y_rx, H_ul_est)):
            raise ValueError("Receiver inputs and parameters must share a device")
        if not all(bool(torch.isfinite(t).all()) for t in (key, shared, Y_rx, H_ul_est)):
            raise ValueError("Receiver complex inputs must be finite")
        gamma = self._scalar(feedback_gamma_db, batch, device, "feedback_gamma_db")
        n0 = self._scalar(noise_variance, batch, device, "noise_variance", allow_zero=False)
        ep = self._scalar(pilot_energy_per_ue, batch, device, "pilot_energy_per_ue", allow_zero=False)
        ek = self._scalar(key_energy_per_ue, batch, device, "key_energy_per_ue")
        es = self._scalar(shared_energy_per_ue, batch, device, "shared_energy_per_ue", allow_zero=False)
        ce_var = torch.as_tensor(channel_error_variance, dtype=torch.float32, device=device)
        if ce_var.ndim == 0:
            ce_var = ce_var.expand(batch, K_USERS)
        if ce_var.shape != (batch, K_USERS) or not bool(torch.isfinite(ce_var).all() & (ce_var >= 0).all()):
            raise ValueError("channel_error_variance must be finite nonnegative [B,8]")
        if not bool(((gamma >= 0) & (gamma <= 25)).all()):
            raise ValueError("feedback_gamma_db must lie in [0,25]")

        key_rel = torch.as_tensor(key_reliability, dtype=torch.float32, device=device)
        shared_rel = torch.as_tensor(shared_reliability, dtype=torch.float32, device=device)
        active_shared = 20 if layout_kind == "fdma" else expected_shared
        if key_rel.shape != (batch, K_USERS, 4) or shared_rel.shape != (batch, K_USERS, active_shared):
            raise ValueError("Reliability shape does not match active received symbols")
        if not bool(torch.isfinite(key_rel).all() & torch.isfinite(shared_rel).all()
                    & (key_rel >= 0).all() & (key_rel <= 1).all()
                    & (shared_rel >= 0).all() & (shared_rel <= 1).all()):
            raise ValueError("Reliabilities must be finite and lie in [0,1]")

        symbols = torch.zeros((batch, K_USERS, MAX_SYMBOLS_PER_USER), dtype=torch.complex64, device=device)
        reliability = torch.zeros((batch, K_USERS, MAX_SYMBOLS_PER_USER), dtype=torch.float32, device=device)
        mask = active.to(device)[None].expand(batch, -1, -1)
        symbols[..., :4] = key
        reliability[..., :4] = key_rel
        if layout_kind == "fdma":
            symbols[..., 4:24] = shared[..., :20]
            reliability[..., 4:24] = shared_rel
            # The first four packed FDMA coordinates are passed through key;
            # relabel every active coordinate as FDMA data in query metadata.
            kind = torch.zeros_like(kind)
            kind[:, :24] = 2
        else:
            symbols[..., 4:4 + expected_shared] = shared
            reliability[..., 4:4 + expected_shared] = shared_rel

        resource_d = resource.to(device)
        role_grid_cpu = torch.zeros((N_SUBCARRIERS, K_USERS * 4), dtype=torch.float32)
        for user, positions in enumerate(layout.pilot_positions):
            role_grid_cpu[torch.as_tensor(positions), user * 4] = 1.0
        rel_grid = torch.zeros((batch, N_SUBCARRIERS, K_USERS), dtype=torch.float32, device=device)
        for user in range(K_USERS):
            slots = torch.nonzero(active[user], as_tuple=False).flatten()
            q_cpu = resource[user, slots]
            role_grid_cpu[q_cpu, user * 4 + kind[user, slots] + 1] = 1.0
            q = resource_d[user, slots.to(device)]
            rel_grid[:, q, user] = reliability[:, user, slots.to(device)]
        role_grid = role_grid_cpu.to(device)
        memory = self._observation_memory(Y_rx, H_ul_est, rel_grid, role_grid,
            gamma, n0, ep, ek, es, group_count, layout_kind, ce_var)
        static, user_ids, group_ids, resource_ids, flat_active = self._query_metadata(
            resource, kind, group, active, reliability, ce_var, batch, device)
        features = torch.cat((torch.view_as_real(symbols.contiguous()),
            torch.log1p(reliability)[..., None], static), dim=-1).reshape(
                batch, K_USERS * MAX_SYMBOLS_PER_USER, 23)
        context = self.query_projection(features) * flat_active[None, :, None]
        coupling = _channel_coupling(H_ul_est)
        self_bias = self._self_bias(user_ids, group_ids, resource_ids, flat_active, coupling)
        obs_bias = self._observation_bias(resource_ids)
        for layer in self.joint_layers:
            context = _run(layer, context, memory, flat_active[None].expand(batch, -1),
                self_bias, obs_bias,
                enabled=self.activation_checkpoint and self.training)
        context1 = context
        context_by_user = context1.reshape(batch, K_USERS, MAX_SYMBOLS_PER_USER, self.width)
        residual = self.residual_head(self.residual_norm(context_by_user))
        correction = torch.complex(residual[..., 0], residual[..., 1]).masked_fill(~mask, 0)
        corrected = (symbols + correction).masked_fill(~mask, 0)
        z_sem = corrected
        def recover_physical(value):
            return residual_ops.regularized_matched_residual(
                value, Y_rx, H_ul_est, resource, kind, active,
                noise_variance=n0, key_energy_per_ue=ek, shared_energy_per_ue=es,
                channel_error_variance=ce_var, pilot_positions=layout.pilot_positions)
        values = self._physical_interface(symbols, context1.reshape(
            batch, K_USERS, MAX_SYMBOLS_PER_USER, self.width), z_sem,
            reliability, mask, recover_physical, bridge_enabled=bridge_enabled,
            physical_from_soft=physical_from_soft)
        corrected = values['z_recon']

        spatial = self.base_decoder.expand_packed_symbols(corrected, mask,
            group_count=group_count, layout_kind=layout_kind)
        spatial = spatial.reshape(batch, K_USERS, 512, self.width)
        position = self.spatial_position(self.spatial_features).unsqueeze(0)
        spatial = spatial + position
        spatial_flat = spatial.reshape(batch, K_USERS * 512, self.width)
        # The original spatial path always reads first-round context1.
        context_keys = self.spatial_context_norm(context1)
        spatial_flat = spatial_flat + self.spatial_attention(
            self.spatial_query_norm(spatial_flat), context_keys,
            key_mask=flat_active[None].expand(batch, -1))
        spatial = spatial_flat.reshape(batch * K_USERS, 512, self.width)
        for block in self.base_decoder.blocks:
            spatial = _run(block, spatial, 64, 8,
                enabled=self.activation_checkpoint and self.training)
        H_hat = self.base_decoder.tokens_to_channel(spatial, batch, normalize=True)
        if not bool(torch.isfinite(H_hat).all() & torch.isfinite(corrected).all()):
            raise FloatingPointError("K8 Joint Transformer receiver produced nonfinite output")
        execution = {
            "layout": layout_kind, "group_count": group_count,
            "active_query_count": int(active.sum()), "padded_query_count": int(active.numel() - active.sum()),
            "observation_memory_shape": list(memory.shape),
            "joint_context_shape": list(context.shape),
            "spatial_cross_attention_shape": [batch, K_USERS * 512, self.width],
            "swin_block_count": len(self.base_decoder.blocks),
            "estimated_channel_coupling_shape": list(coupling.shape),
        }
        execution.update(variant=self.variant, architecture=ARCHITECTURE,
            joint_layer_count=2, joint_layer_executions=2,
            spatial_decoder_executions=1, spatial_context='context1',
            physical_head_executions=1, bridge_enabled=values['bridge_enabled'],
            physical_from_soft=values['physical_from_soft'],
            bridge_executions=int(values['bridge_enabled']))
        expose=self.include_stages if return_stages is None else return_stages
        if type(expose) is not bool:raise ValueError('return_stages must be a boolean')
        if expose:
            execution['stages']=dict(soft=values['soft'],x_phy=values['x_phy'],
                z_sem=z_sem,z_recon=corrected,d_phy=values['d_phy'],reliability=reliability,
                active_mask=active.to(device),resource_map=resource_d,kind=kind.to(device),
                rx_residual_physical_mean_power_unnormalized=values['residual_energy'],
                bridge_enabled=values['bridge_enabled'],physical_from_soft=values['physical_from_soft'])
        return H_hat, corrected, execution
