"""Isolated base2/deep4/DC2 variants of the unchanged K8 BS receiver.

The original receiver's validation, packing and spatial path are retained in
this explicit forward; no hooks, hidden forward caches or simulator truth are
used.  Only the declared new module parameters differ from a source codec.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

from models.twc_v1_k8_joint_decoder import (
    K8JointTransformerSwinReceiver, _JointLayer, _run, _channel_coupling,
    K_USERS, N_ANTENNAS, N_SUBCARRIERS, MAX_SYMBOLS_PER_USER, HEADS, QUERY_CHUNK,
)
from models.twc_v1_k8_ordered_encoder import build_codec as build_ordered_codec

_RESIDUAL_NAME='_twc_v1_k8_decoder_residual'
_RESIDUAL_PATH=Path(__file__).resolve().parents[1]/'utils/twc_v1_k8_decoder_residual.py'
if _RESIDUAL_NAME in sys.modules:
    residual_ops=sys.modules[_RESIDUAL_NAME]
    if Path(residual_ops.__file__).resolve()!=_RESIDUAL_PATH.resolve():
        raise RuntimeError('Decoder residual module identity collision')
else:
    _spec=importlib.util.spec_from_file_location(_RESIDUAL_NAME,_RESIDUAL_PATH)
    if _spec is None or _spec.loader is None:raise RuntimeError('Cannot load decoder residual operators')
    residual_ops=importlib.util.module_from_spec(_spec);sys.modules[_RESIDUAL_NAME]=residual_ops
    try:_spec.loader.exec_module(residual_ops)
    except BaseException:
        del sys.modules[_RESIDUAL_NAME]
        raise

VARIANTS=('base2','deep4','dc2')
ARCHITECTURE='twc_v1_k8_decoder_refinement_v1'
DC2_EXPECTED_NEW_PARAMETERS=514
_ATTN_KEYS=tuple(name+'.'+field for name in ('q','k','v','out') for field in ('weight','bias'))
_LAYER_KEYS=(tuple(name+'.'+field for name in ('self_norm','cross_norm','memory_norm','mlp_norm')
                   for field in ('weight','bias'))
    +tuple(name+'.'+key for name in ('self_attention','cross_attention') for key in _ATTN_KEYS)
    +tuple('mlp.'+name+'.'+field for name in ('0','2') for field in ('weight','bias')))


def added_state_keys(variant, *, prefix='joint_decoder.'):
    if variant not in VARIANTS:raise ValueError('One explicit decoder variant is required')
    if variant=='base2':return ()
    if variant=='dc2':return tuple(prefix+key for key in (
        'residual_projection.weight','second_residual_head.weight','second_residual_head.bias'))
    return tuple(prefix+'joint_layers.'+str(index)+'.'+key for index in (2,3) for key in _LAYER_KEYS)


def initialize_identity_joint_layer(layer):
    """Zero only the three residual output projections of an added layer."""
    with torch.no_grad():
        for output in (layer.self_attention.out,layer.cross_attention.out,layer.mlp[-1]):
            output.weight.zero_();output.bias.zero_()
    return layer


def make_dc_residual_modules(width=128):
    """The sole DC2 additions: 2->D without bias and zero D->2 output."""
    if type(width) is not int or width<1:raise ValueError('A positive residual width is required')
    projection=nn.Linear(2,width,bias=False)
    head=nn.Linear(width,2)
    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    return projection,head


def migrate_source_state(model,source_state):
    """Explicit strict source migration; ordinary resume must use strict load.

    The complete old key set is required.  New keys are exactly the variant's
    declared whitelist.  Conflicting values for a shared decoder alias are
    rejected before load, since load order must not silently choose a value.
    """
    receiver=model.joint_decoder
    new=set(added_state_keys(receiver.variant));template=model.state_dict()
    if not isinstance(source_state,dict) or set(source_state)!=set(template)-new:
        actual=set(source_state) if isinstance(source_state,dict) else set()
        raise ValueError('Source state key mismatch; missing='+str(sorted((set(template)-new)-actual))+
                         '; unexpected='+str(sorted(actual-(set(template)-new))))
    if not new<=set(template):raise ValueError('Target does not contain the declared new modules')
    converted={};aliases={}
    for name,expected in template.items():
        value=expected if name in new else source_state[name]
        if (not torch.is_tensor(value) or value.shape!=expected.shape or value.dtype!=expected.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Source tensor shape/dtype/finite mismatch: '+name)
        storage=(expected.device,expected.untyped_storage().data_ptr(),expected.storage_offset(),tuple(expected.shape),tuple(expected.stride()),expected.dtype)
        if name not in new and storage in aliases and not torch.equal(value,source_state[aliases[storage]]):
            raise ValueError('Conflicting shared decoder aliases: '+aliases[storage]+' and '+name)
        if name not in new:aliases.setdefault(storage,name)
        converted[name]=value.detach().clone()
    model.load_state_dict(converted,strict=True)
    loaded=model.state_dict()
    for name,value in source_state.items():
        if not torch.equal(loaded[name].detach().cpu(),value.detach().cpu()):
            raise RuntimeError('Strict migration altered source tensor: '+name)
    new_parameters=sum(p.numel() for name,p in model.named_parameters() if name in new)
    return dict(strict=True,copied_old_keys=len(source_state),added_keys=sorted(new),
                new_parameters=new_parameters,source_values_preserved=True)


def build_codec(variant='base2', *, new_module_seed=20261107, include_stages=True, **kwargs):
    """Keep the original encoder/FC/Swin and replace only the BS receiver."""
    if variant not in VARIANTS:raise ValueError('Unknown decoder refinement variant')
    codec=build_ordered_codec(ordered=True,**kwargs)
    codec.joint_decoder=K8DecoderRefinementReceiver(codec.decoder,variant=variant,
        new_module_seed=new_module_seed,include_stages=include_stages)
    if codec.joint_decoder.base_decoder is not codec.decoder:raise RuntimeError('Shared spatial decoder alias was lost')
    return codec


class K8DecoderRefinementReceiver(K8JointTransformerSwinReceiver):
    def __init__(self,base_decoder,*,variant='base2',new_module_seed=20261107,
                 include_stages=True,query_chunk=QUERY_CHUNK,activation_checkpoint=True):
        if variant not in VARIANTS or type(include_stages) is not bool:
            raise ValueError('Explicit decoder variant and boolean stage option required')
        if type(new_module_seed) is not int or not 0<=new_module_seed<2**63:
            raise ValueError('An explicit bounded integer new-module seed is required')
        # Local initialization cannot perturb the UE or other modules' RNG stream.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(new_module_seed)
            super().__init__(base_decoder,joint_layers=2,query_chunk=query_chunk,
                             activation_checkpoint=activation_checkpoint)
            if variant=='deep4':
                for _ in range(2):
                    self.joint_layers.append(initialize_identity_joint_layer(_JointLayer(
                        width=self.width,query_chunk=query_chunk,activation_checkpoint=activation_checkpoint)))
            elif variant=='dc2':
                self.residual_projection,self.second_residual_head=make_dc_residual_modules(self.width)
        self.variant=variant;self.include_stages=include_stages;self.new_module_seed=new_module_seed

    def _local_self_bias(self,user_ids,resource_ids,active,original_bias):
        def build(start,end):
            allowed=residual_ops.local_attention_allowed(user_ids,resource_ids,active,start=start,end=end)
            return original_bias(start,end).masked_fill(~allowed[None,None],-torch.inf)
        return build

    def forward(self, key, shared, *, group_count, Y_rx, H_ul_est, layout,
                feedback_gamma_db, noise_variance, pilot_energy_per_ue,
                key_energy_per_ue, shared_energy_per_ue, channel_error_variance,
                key_reliability, shared_reliability, return_stages=None):
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
        x1 = corrected
        residual_info = None
        if self.variant == 'dc2':
            residual_info = residual_ops.regularized_matched_residual(
                x1, Y_rx, H_ul_est, resource, kind, active,
                noise_variance=n0, key_energy_per_ue=ek, shared_energy_per_ue=es,
                channel_error_variance=ce_var, pilot_positions=layout.pilot_positions)
            delta_features = torch.view_as_real(residual_info['correction'].contiguous()).reshape(
                batch, K_USERS * MAX_SYMBOLS_PER_USER, 2)
            features2 = torch.cat((torch.view_as_real(x1.contiguous()),
                torch.log1p(reliability)[..., None], static), dim=-1).reshape(
                    batch, K_USERS * MAX_SYMBOLS_PER_USER, 23)
            context2 = (self.query_projection(features2) + self.residual_projection(delta_features))
            context2 = context2 * flat_active[None, :, None]
            local_bias = self._local_self_bias(user_ids, resource_ids, flat_active, self_bias)
            for layer in self.joint_layers:
                context2 = _run(layer, context2, memory, flat_active[None].expand(batch, -1),
                    local_bias, obs_bias, enabled=self.activation_checkpoint and self.training)
            delta2 = self.second_residual_head(self.residual_norm(context2.reshape(
                batch, K_USERS, MAX_SYMBOLS_PER_USER, self.width)))
            delta2 = torch.complex(delta2[..., 0], delta2[..., 1]).masked_fill(~mask, 0)
            corrected = (x1 + delta2).masked_fill(~mask, 0)

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
        execution.update(variant=self.variant, refinement_rounds=2 if self.variant=='dc2' else 1,
            joint_layer_count=len(self.joint_layers),
            joint_layer_executions=len(self.joint_layers)*(2 if self.variant=='dc2' else 1),
            spatial_decoder_executions=1, spatial_context='context1')
        expose=self.include_stages if return_stages is None else return_stages
        if type(expose) is not bool:raise ValueError('return_stages must be a boolean')
        if expose:
            pilots=layout.pilot_positions
            energy1=(residual_info['residual_energy'] if residual_info is not None else
                residual_ops.data_received_residual(x1,Y_rx,H_ul_est,resource,active,
                    pilot_positions=pilots)['energy'])
            final_energy=(residual_ops.data_received_residual(corrected,Y_rx,H_ul_est,resource,
                active,pilot_positions=pilots)['energy'] if self.variant=='dc2' else energy1)
            execution['stages']=dict(soft=symbols,x1=x1,x2=corrected,
                active_mask=active.to(device),resource_map=resource_d,kind=kind.to(device),
                received_residual_energy_x1=energy1,received_residual_energy_final=final_energy,
                second_round=self.variant=='dc2')
        return H_hat, corrected, execution
