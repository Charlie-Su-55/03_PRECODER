#!/usr/bin/env python3
"""Server-only synthetic acceptance of the new hybrid/prior lifecycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import traceback

from configs.grouped_hybrid_anchor_train_config import ARMS, SEEDS, OUTPUT_ROOT, anchor_config, protocol
from as_receiver_batch_support import safe_inside, sha256, save_json
from preflight_proposal_as_direct import _backend_info
from preflight_joint_swin import tensor_digest
from preflight_grouped_codec import required_modules, POWER_ATOL

ROOT = Path(__file__).resolve().parent


def plan():
    return dict(arms=list(ARMS), seeds=list(SEEDS), batch_size=48, cuda_required=True,
        synthetic_channels=True, old_weights='same-seed pinned F best; initialization only',
        gamma_db=12.5, downlink_snr_db=25, power_atol=POWER_ATOL,
        parameter_counts='measured on server, never inferred as measured locally',
        official_optimizer_updates=0, scratch_updates_per_arm=1,
        physical_budget='4*Df+2*Da=256', energy_per_ue=64,
        prior='forward reads committed history; scratch optimizer success precedes commit',
        automatic_fallback=False, no_saved_weights=True)


def initialization(torch, report):
    from train_grouped_hybrid_anchor import load_source, validate_model_identity
    from models.grouped_hybrid_anchor import GroupedHybridAnchorPrecoder, migrate_legacy_fdma_state_dict
    report['initialization'] = {}
    for seed in SEEDS:
        source, source_record = load_source(torch, seed)
        reference = None
        records = {}
        for arm in ARMS:
            before = torch.get_rng_state().clone()
            cfg = anchor_config(arm, seed)
            model = GroupedHybridAnchorPrecoder(cfg)
            mapping = migrate_legacy_fdma_state_dict(model, source['state'])
            validate_model_identity(cfg, model)
            model.power_prior.validate_state(0)
            model.load_state_dict(model.state_dict(), strict=True)
            hashes = {name: tensor_digest({'tensor':value}) for name,value in model.named_parameters()}
            if reference is None: reference = hashes
            elif hashes != reference: raise ValueError('Same-seed arms must have identical parameter initialization')
            if not torch.equal(before, torch.get_rng_state()): raise ValueError('Construction changed external CPU RNG')
            counts = dict(ue=sum(p.numel() for p in model.ue_parameters()),
                bs=sum(p.numel() for p in model.bs_parameters()), full=sum(p.numel() for p in model.parameters()),
                added_anchor=sum(p.numel() for p in model.decoder.anchor.parameters()))
            records[arm] = dict(parameters=counts,parameter_hashes=hashes,migration=mapping,
                strict_roundtrip=True,committed_prior_updates=0,external_rng_unchanged=True)
            del model
        if len({json.dumps(row['parameters'],sort_keys=True) for row in records.values()}) != 1:
            raise ValueError('Arm parameter capacities differ')
        report['initialization'][str(seed)] = dict(source=source_record,arms=records,
            all_parameters_paired=True,source_forward_equivalence_claimed=False)
        del source


def receiver_boundary(torch, model):
    import inspect
    expected = ('z_hat','error_feature','Y','H_ul_est','gamma_db')
    if tuple(inspect.signature(model.decoder.forward).parameters) != expected:
        raise ValueError('Receiver public input boundary changed')
    model.eval()
    with torch.no_grad():
        soft=torch.randn(1,4,256,device='cuda',dtype=torch.complex64)
        error=torch.full_like(soft.real,.3)
        anchor=model.decoder.anchor
        geometry=model.decoder.query_fixed;mask=model.decoder.exclusive_mask
        base=anchor(soft,error,geometry,mask)
        ignored=soft.clone();ignored[:,~mask]+=torch.complex(torch.tensor(9.,device='cuda'),torch.tensor(4.,device='cuda'))
        if not torch.equal(base,anchor(ignored,error,geometry,mask)):
            raise ValueError('Anchor read shared/inactive symbols')
        changed=soft.clone();changed[:,0,mask[0]]+=.5+.2j
        affected=anchor(changed,error,geometry,mask)
        if torch.equal(base[:,0],affected[:,0]) or not torch.equal(base[:,1:],affected[:,1:]):
            raise ValueError('Own exclusive observations must affect only that user anchor')
    return dict(decoder_inputs=list(expected),shared_and_inactive_anchor_isolation=True,
        own_user_exclusive_dependency=True,truth_not_decoder_input=True,
        model_error_feature='log1p(linear_error_proxy / historical_power_prior)')


def batch48(torch, arm, report):
    from train_grouped_hybrid_anchor import load_source, AnchorLifecycle, validate_model_state
    from models.grouped_hybrid_anchor import GroupedHybridAnchorPrecoder, migrate_legacy_fdma_state_dict
    from main_train import SumRateLoss, set_seed
    cfg=anchor_config(arm,SEEDS[0]);set_seed(protocol.namespace_seed(cfg.SEED,'initialization'))
    source,_=load_source(torch,cfg.SEED)
    model=GroupedHybridAnchorPrecoder(cfg);migrate_legacy_fdma_state_dict(model,source['state']);del source
    model=model.cuda().train()
    result=report.setdefault('batch48',{}).setdefault(arm,dict(passed=False))
    torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    lifecycle=AnchorLifecycle(cfg)
    hdl=torch.randn(48,4,32,256,device='cuda',dtype=torch.complex64);hul=torch.randn_like(hdl)
    noise=protocol.standard_noise(cfg.SEED,'preflight',0,48,device='cuda')
    modules=dict(model.named_modules())
    names=required_modules()+['decoder.anchor','ue.condition','ue.encoder.aircomp_backbone','ue.encoder.fdma_backbone']
    calls={name:0 for name in names};handles=[]
    def counter(name):
        def hook(*unused):calls[name]+=1
        return hook
    before={name:value.clone() for name,value in model.power_prior.state_dict().items()}
    try:
        for name in names: handles.append(modules[name].register_forward_hook(counter(name)))
        output=model.training_with_statistics(hdl,hul,12.5,global_step=1,standard_noise=noise)
        result['execution_counts']=dict(calls)
        if calls['ue.encoder.fdma_backbone'] or any(calls[n]<1 for n in names if n!='ue.encoder.fdma_backbone'):
            raise ValueError('Full Joint/anchor/AS path did not execute as required')
        if any(not torch.equal(value,model.power_prior.state_dict()[name]) for name,value in before.items()):
            raise ValueError('Forward changed historical prior')
        w,hat,pending=output
        if tuple(hat.shape)!=(48,4,32,256) or pending['read_count']!=0:
            raise ValueError('Incorrect complete channel or prior read step')
        deployed=model.zf_solver(hat)
        deployed=deployed*torch.sqrt(1/(deployed.abs().square().sum((1,2),keepdim=True)+1e-9))
        loss,rate=SumRateLoss()(deployed,hdl,10**(-25/10))
        result.update(power_max_abs=(deployed.abs().square().sum((1,2))-1).abs().max().item(),
            normalized_energy_max_abs=(pending['power_mean'].sum(-1)-64).abs().max().item(),
            synthetic_rate_not_performance_result=rate)
        if result['power_max_abs']>POWER_ATOL or not torch.isfinite(loss):
            raise FloatingPointError('Task loss/power invalid')
        loss.backward();result['gradients']={}
        for name in names:
            grads=[p.grad for p in modules[name].parameters() if p.grad is not None]
            norm=sum(g.detach().double().square().sum().item() for g in grads)**.5
            finite=all(torch.isfinite(g).all().item() for g in grads)
            result['gradients'][name]=dict(norm=norm,finite=finite,tensors=len(grads))
            if name=='ue.encoder.fdma_backbone':
                if grads: raise ValueError('Stored inactive FDMA encoder executed')
            elif not grads or not finite or norm<=0: raise ValueError('Broken active gradient '+name)
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
        optimizer.step()
        lifecycle.after_optimizer_step(model,output,1);lifecycle.validate_model_step(model,1)
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite scratch update')
        result.update(prior_unchanged_before_optimizer=True,committed_update_count=1,
            node_update_mass=model.power_prior.node_update_mass.tolist())
        torch.cuda.synchronize()
    finally:
        for handle in handles:handle.remove()
        result.update(hooks_removed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del output,w,hat,pending,deployed,loss,optimizer,hdl,hul,noise
    model.zero_grad(set_to_none=True);model.eval()
    state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    validate_model_state(torch,model,state,step=1);model.load_state_dict(state,strict=True)
    bad=dict(state);bad.pop('power_prior.pi_nodes')
    try:model.load_state_dict(bad,strict=True)
    except RuntimeError:pass
    else:raise ValueError('Missing prior silently restored')
    model.load_state_dict(state,strict=True)
    before={name:value.clone() for name,value in model.named_buffers()}
    with torch.no_grad():
        h=torch.randn(1,4,32,256,device='cuda',dtype=torch.complex64)
        e=protocol.standard_noise(cfg.SEED,'preflight',1,1,device='cuda')
        diagnostic=model.forward_with_diagnostics(h,torch.randn_like(h),12.5,standard_noise=e)['diagnostics']
        if (diagnostic['energy']-64).abs().max().item()>1e-4:raise ValueError('Per-user energy changed')
        if diagnostic['N0']!=protocol.noise_variance(12.5):raise ValueError('Noise convention changed')
    result['receiver_boundary']=receiver_boundary(torch,model)
    if any(not torch.equal(value,dict(model.named_buffers())[name]) for name,value in before.items()):
        raise ValueError('Evaluation changed buffer/prior')
    result.update(strict_prior_restore=True,missing_prior_rejected=True,evaluation_prior_frozen=True,passed=True)
    del state,model,diagnostic,h,e
    torch.cuda.empty_cache()


def run(output):
    output=safe_inside(output,OUTPUT_ROOT/'batches');output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,plan=plan())
    try:
        import torch
        from train_grouped_hybrid_anchor import apply_backend,SOURCE_FILES
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        apply_backend(torch)
        report.update(backend=_backend_info(torch),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha256(ROOT/p) for p in (*SOURCE_FILES,'preflight_grouped_hybrid_anchor.py')})
        initialization(torch,report)
        for arm in ARMS:batch48(torch,arm,report)
        report['passed']=True
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),automatic_fallback=False)
        raise
    finally:save_json(output/'report.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args(argv)
    if args.output_dir is not None and not args.run:parser.error('--output-dir requires --run')
    if args.dry_run:print(json.dumps(plan(),indent=2))
    elif args.run:
        if args.output_dir is None:parser.error('Explicit unused --output-dir required')
        run(args.output_dir)
    else:parser.print_help()
    return 0


if __name__=='__main__':raise SystemExit(main())
