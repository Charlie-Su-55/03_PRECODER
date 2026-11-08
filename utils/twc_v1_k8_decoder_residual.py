"""Receiver-only physical residual features and separate offline diagnostics.

No channel generator, model, package initializer or simulator truth is imported.
Decoder features consume noisy observations, estimated UL CSI and public layout
and power information only.  ``offline_decoder_diagnostics`` is metric-only.
"""
from __future__ import annotations

import torch


def packed_layout_map(layout, *, users=8, slots=164):
    """Match the original receiver's user-major physical-data packing."""
    resource=torch.zeros(users,slots,dtype=torch.long)
    kind=torch.zeros_like(resource)
    group=torch.full_like(resource,-1)
    active=torch.zeros(users,slots,dtype=torch.bool)
    if hasattr(layout,'shared_positions_by_group'):
        user_group={u:g for g,members in enumerate(layout.groups) for u in members}
        for user,positions in enumerate(layout.key_positions):
            resource[user,:4]=torch.as_tensor(positions)
            group[user,:4]=user_group[user];active[user,:4]=True
        width=layout.shared_dimension
        for gi,(members,positions) in enumerate(zip(layout.groups,layout.shared_positions_by_group)):
            for user in members:
                resource[user,4:4+width]=torch.as_tensor(positions)
                kind[user,4:4+width]=1;group[user,4:4+width]=gi;active[user,4:4+width]=True
        return resource,kind,group,active,'hybrid',layout.group_count
    if hasattr(layout,'data_positions'):
        for user,positions in enumerate(layout.data_positions):
            n=len(positions);resource[user,:n]=torch.as_tensor(positions)
            kind[user,:n]=2;active[user,:n]=True
        return resource,kind,group,active,'fdma',1
    raise TypeError('An explicit K8 HybridLayout or FDMALayout is required')


def _vector(value,batch,device,name,*,positive=False):
    result=torch.as_tensor(value,dtype=torch.float32,device=device)
    if result.ndim==0:result=result.expand(batch)
    sign=result>0 if positive else result>=0
    if result.shape!=(batch,) or not bool(torch.isfinite(result).all() & sign.all()):
        raise ValueError(name+' must be a finite valid scalar or [B]')
    return result


def _metadata(symbols,Y_rx,H_ul_est,resource,active,pilot_positions):
    if (not torch.is_tensor(symbols) or symbols.ndim!=3 or symbols.dtype!=torch.complex64
            or not torch.is_tensor(Y_rx) or Y_rx.ndim!=3 or Y_rx.dtype!=symbols.dtype
            or not torch.is_tensor(H_ul_est) or H_ul_est.ndim!=4 or H_ul_est.dtype!=symbols.dtype):
        raise ValueError('Physical symbols/Y/H_est must be complex64 [B,K,S]/[B,Nt,Q]/[B,K,Nt,Q]')
    batch,users,slots=symbols.shape
    if (batch<1 or tuple(H_ul_est.shape)!=(batch,users,Y_rx.shape[1],Y_rx.shape[2])
            or Y_rx.shape[0]!=batch or resource.shape!=(users,slots) or active.shape!=resource.shape
            or resource.dtype!=torch.long or active.dtype!=torch.bool):
        raise ValueError('Physical residual axes or packed metadata do not agree')
    if len({symbols.device,Y_rx.device,H_ul_est.device})!=1:
        raise ValueError('Receiver tensors must share a device')
    if not all(bool(torch.isfinite(v).all()) for v in (symbols,Y_rx,H_ul_est)):
        raise ValueError('Physical residual inputs must be finite')
    # Layout metadata is not a learned feature.  Build collision groups on CPU
    # rather than synchronizing one CUDA scalar for every packed coordinate.
    resource=resource.detach().cpu();active=active.detach().cpu()
    if not bool(active.any()) or bool(((resource[active]<0)|(resource[active]>=Y_rx.shape[-1])).any()):
        raise ValueError('At least one valid physical data coordinate is required')
    pilots={int(q) for row in pilot_positions for q in row} if pilot_positions else set()
    for user in range(users):
        q=resource[user,active[user]]
        if q.numel()!=torch.unique(q).numel():raise ValueError('A UE cannot transmit twice on the same data RE')
        if any(int(v) in pilots for v in q.tolist()):
            raise ValueError('Pilot REs cannot enter the data residual')
    return resource,active


def data_received_residual(symbols,Y_rx,H_ul_est,resource,active,*,pilot_positions=()):
    """Y_data - H_est*x at the unchanged physical normalized-symbol scale.

    Values on inactive/pilot REs are zeroed; symbols are never renormalized.
    The energy is mean squared magnitude over antennas and active data REs.
    """
    resource,active=_metadata(symbols,Y_rx,H_ul_est,resource,active,pilot_positions)
    prediction=torch.zeros_like(Y_rx)
    data_mask=torch.zeros(Y_rx.shape[-1],dtype=torch.bool,device=Y_rx.device)
    for user in range(symbols.shape[1]):
        slots=torch.nonzero(active[user],as_tuple=False).flatten()
        q=resource[user,slots].to(Y_rx.device)
        contribution=H_ul_est[:,user].index_select(-1,q)*symbols[:,user,slots.to(Y_rx.device)][:,None]
        prediction=prediction.index_add(-1,q,contribution)
        data_mask[q]=True
    residual=(Y_rx-prediction)*data_mask[None,None]
    data_count=torch.unique(resource[active]).numel()
    energy=residual.abs().square().sum((1,2))/(data_count*Y_rx.shape[1])
    if not bool(torch.isfinite(residual).all() & torch.isfinite(energy).all()):
        raise FloatingPointError('Nonfinite physical data residual')
    return dict(residual=residual,energy=energy,data_mask=data_mask)


def solve_matched_residual(channel,residual,ridge):
    """Solve (G^H G + ridge I)d=G^H r, with explicit complex axes.

    G [...,Nt,U], r [...,Nt,1], ridge broadcastable to G's leading axes.
    Complex128 is allowed for tiny algebra verification; production is FP32.
    """
    if (not torch.is_tensor(channel) or channel.dtype not in (torch.complex64,torch.complex128)
            or residual.dtype!=channel.dtype or channel.ndim<2
            or residual.shape!=(*channel.shape[:-1],1)
            or not bool(torch.isfinite(channel).all() & torch.isfinite(residual).all())):
        raise ValueError('Finite complex channel [...,Nt,U] and residual [...,Nt,1] required')
    ridge=torch.as_tensor(ridge,dtype=channel.real.dtype,device=channel.device)
    if not bool(torch.isfinite(ridge).all() & (ridge>0).all()):raise ValueError('Positive finite ridge required')
    gh=channel.conj().transpose(-2,-1)
    eye=torch.eye(channel.shape[-1],dtype=channel.dtype,device=channel.device)
    solved=torch.linalg.solve(gh@channel+ridge[...,None,None]*eye,gh@residual)
    if not bool(torch.isfinite(solved).all()):raise FloatingPointError('Nonfinite matched residual solve')
    return solved.squeeze(-1)


def regularized_matched_residual(symbols,Y_rx,H_ul_est,resource,kind,active,*,
        noise_variance,key_energy_per_ue,shared_energy_per_ue,channel_error_variance,pilot_positions=()):
    """Original receiver ridge convention on actual colliding data streams.

    Key: N0/(Ek/4)+ce_user; shared: N0/(Es/a)+sum(ce_group);
    FDMA: N0/(Edata/24)+ce_user.  Each retains the original 1e-8 floor.
    This is a preconditioned residual feature, not a calibrated posterior.
    """
    received=data_received_residual(symbols,Y_rx,H_ul_est,resource,active,pilot_positions=pilot_positions)
    resource=resource.detach().cpu();active=active.detach().cpu();kind=kind.detach().cpu()
    if kind.shape!=active.shape or kind.dtype!=torch.long or bool(((kind[active]<0)|(kind[active]>2)).any()):
        raise ValueError('Packed active kinds must be key/shared/FDMA IDs 0/1/2')
    batch,users,_=symbols.shape;device=symbols.device
    n0=_vector(noise_variance,batch,device,'noise variance',positive=True)
    ek=_vector(key_energy_per_ue,batch,device,'key energy')
    es=_vector(shared_energy_per_ue,batch,device,'shared/data energy',positive=True)
    ce=torch.as_tensor(channel_error_variance,dtype=torch.float32,device=device)
    if ce.ndim==0:ce=ce.expand(batch,users)
    if ce.shape!=(batch,users) or not bool(torch.isfinite(ce).all() & (ce>=0).all()):
        raise ValueError('CE error proxy must be finite nonnegative [B,K]')
    by_resource={}
    for user in range(users):
        for slot in torch.nonzero(active[user],as_tuple=False).flatten().tolist():
            q=int(resource[user,slot]);by_resource.setdefault(q,[]).append((user,slot,int(kind[user,slot])))
    patterns={}
    for q,entries in sorted(by_resource.items()):
        entry_kind={v[2] for v in entries}
        if len(entry_kind)!=1:raise ValueError('A physical data RE cannot mix key/shared/FDMA kinds')
        signature=(tuple(v[0] for v in entries),next(iter(entry_kind)))
        patterns.setdefault(signature,[]).append((q,tuple(v[1] for v in entries)))
    correction=torch.zeros_like(symbols)
    for (members,coordinate_kind),entries in patterns.items():
        q=torch.tensor([v[0] for v in entries],dtype=torch.long,device=device)
        user_index=torch.tensor(members,dtype=torch.long,device=device)
        if coordinate_kind!=1 and len(members)!=1:raise ValueError('Only shared REs may collide across users')
        per_user=[int(((kind[u]==coordinate_kind)&active[u]).sum()) for u in members]
        if len(set(per_user))!=1 or per_user[0]<=0:raise ValueError('Nominal per-stream width is inconsistent')
        power=(ek if coordinate_kind==0 else es)/per_user[0]
        if not bool((power>0).all()):raise ValueError('An active stream requires positive nominal symbol power')
        ridge=n0/power+ce.index_select(1,user_index).sum(1)+1e-8
        channel=H_ul_est.index_select(1,user_index).index_select(-1,q).permute(0,3,2,1)
        residual=received['residual'].index_select(-1,q).transpose(1,2).unsqueeze(-1)
        delta=solve_matched_residual(channel,residual,ridge[:,None])
        for local,user in enumerate(members):
            slots=torch.tensor([v[1][local] for v in entries],dtype=torch.long,device=device)
            correction[:,user,slots]=delta[:,:,local]
    return dict(correction=correction,residual_energy=received['energy'])


def local_attention_allowed(user_ids,resource_ids,active,*,start=0,end=None):
    """Same user OR actual same-RE collision, never a group_id=-1 shortcut."""
    end=user_ids.numel() if end is None else end
    if (user_ids.ndim!=1 or resource_ids.shape!=user_ids.shape or active.shape!=user_ids.shape
            or active.dtype!=torch.bool or not 0<=start<end<=user_ids.numel()):
        raise ValueError('Flat user/resource/active metadata required')
    allowed=((user_ids[start:end,None]==user_ids[None]) |
             (resource_ids[start:end,None]==resource_ids[None]))
    allowed=allowed & active[start:end,None] & active[None]
    if bool((active[start:end] & ~allowed.any(-1)).any()):raise ValueError('An active query has no permitted key')
    return allowed


def offline_decoder_diagnostics(output):
    """Metric-only errors against actual normalized transmitted symbols.

    Call only after the deployed forward.  These truth fields never enter any
    decoder feature/forward function in this module.
    """
    resource,kind,_,active,layout_kind,_=packed_layout_map(output['layout'])
    tx=output['tx'];soft=output['soft_symbols'];corrected=output['corrected_symbols']
    if corrected is None:raise ValueError('Explicit corrected codewords required for offline diagnostics')
    truth=torch.zeros_like(corrected);initial=torch.zeros_like(corrected)
    if layout_kind=='fdma':
        truth[...,:24]=tx['data_symbols_tx'];initial[...,:24]=soft['data_symbols_hat']
    else:
        width=tx['shared_symbols_tx'].shape[-1]
        truth[...,:4]=tx['key_symbols_tx'];truth[...,4:4+width]=tx['shared_symbols_tx']
        initial[...,:4]=soft['key_symbols_hat'];initial[...,4:4+width]=soft['shared_symbols_hat']
    stages=output.get('receiver_execution',{}).get('stages',{})
    first=stages.get('x1',corrected);final=stages.get('x2',corrected)
    packed={'soft':stages.get('soft',initial),'round1':first,'final':final}
    result={};active=active.to(truth.device);kind=kind.to(truth.device)
    for stage,symbols in packed.items():
        if symbols.shape!=truth.shape or symbols.dtype!=truth.dtype or not bool(torch.isfinite(symbols).all()):
            raise ValueError('Offline packed codeword stage has invalid shape/dtype/value')
        for label,mask in [('all',active)]+[(label,active&(kind==index)) for label,index in (('key',0),('shared',1),('fdma',2))]:
            if not bool(mask.any()):continue
            numerator=((symbols-truth).abs().square()*mask[None]).sum((1,2))
            denominator=(truth.abs().square()*mask[None]).sum((1,2)).clamp_min(1e-20)
            result[f'stage_{stage}_codeword_nmse_{label}']=numerator/denominator
    for label,symbols in (('x1',first),('final',final)):
        name='received_residual_energy_'+label
        result[name]=stages[name] if name in stages else data_received_residual(symbols,output['Y_rx'],
            output['H_ul_est'],resource,active,pilot_positions=output['layout'].pilot_positions)['energy']
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError('Nonfinite offline codeword diagnostic')
    return result
