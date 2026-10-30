#!/usr/bin/env python3
"""Explicit server-only gate acceptance; no formal training or saved weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import traceback

from configs.grouped_hybrid_power_train_config import ARMS,SEEDS,OUTPUT_ROOT,power_config,protocol
from configs.grouped_hybrid_anchor_train_config import anchor_config
from as_receiver_batch_support import safe_inside,sha256,save_json
from preflight_proposal_as_direct import _backend_info
from preflight_joint_swin import tensor_digest
from preflight_grouped_codec import POWER_ATOL

ROOT=Path(__file__).resolve().parent
RTOL=ATOL=1e-6


def plan():
    return dict(arms=list(ARMS),seeds=list(SEEDS),cuda_required=True,physical_batch_size=48,
        synthetic_channels_only=True,synthetic_seed=20270104,zero_gate_samples=2,zero_gate_gamma=[0,12.5,25],
        equality_rtol=RTOL,equality_atol=ATOL,energy_per_ue=64,
        source_prior_offset=20000,extra_gate_parameters=dict(zip(ARMS,(0,0,1,6))),
        official_updates=0,scratch_optimizer_updates_per_arm=1,weights_written=False,
        no_automatic_precision_or_batch_fallback=True)


def _source_files():
    result={}
    for seed in SEEDS:
        for arm in ('fdma_control','hybrid_f32_a64'):
            cfg=anchor_config(arm,seed)
            for attr in ('BEST_PATH','SAVE_PATH','CONFIG_PATH','STATUS_PATH'):
                path=Path(getattr(cfg,attr));result[str(path)]=sha256(path)
    return result


def _compare(torch,a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Invalid paired output shape/dtype/finite state')
    diff=(a-b).abs()
    result=dict(max_abs=diff.max().item(),max_rel=(diff/b.abs().clamp_min(1e-30)).max().item(),
        rtol=RTOL,atol=ATOL,passed=torch.allclose(a,b,rtol=RTOL,atol=ATOL))
    return result


def _parameters(model):
    return dict(ue=sum(p.numel() for p in model.ue_parameters()),
        bs=sum(p.numel() for p in model.bs_parameters()),full=sum(p.numel() for p in model.parameters()),
        gate=sum(p.numel() for p in model.power_gate.parameters()))


def initialization(torch,report):
    from train_grouped_hybrid_power import load_source
    from models.grouped_hybrid_power import GroupedHybridPowerPrecoder,migrate_source_state_dict
    report['initialization']={}
    for seed in SEEDS:
        rows={};common_h32=None;original_counts=None;construction_reference=None
        for arm in ARMS:
            source,record=load_source(torch,arm,seed)
            before=torch.get_rng_state().clone()
            model=GroupedHybridPowerPrecoder(power_config(arm,seed))
            constructed=tensor_digest({k:v for k,v in model.named_parameters() if k!='power_gate.theta'})
            if construction_reference is None:construction_reference=constructed
            elif constructed!=construction_reference:raise ValueError('Gate perturbed shared parameter initialization')
            mapping=migrate_source_state_dict(model,source['state'])
            if not torch.equal(before,torch.get_rng_state()): raise ValueError('Construction changed external CPU RNG')
            model.power_prior.validate_state(20000)
            model.load_state_dict(model.state_dict(),strict=True)
            copied={k:v for k,v in model.state_dict().items() if k!='power_gate.theta'}
            digest=tensor_digest(copied)
            counts=_parameters(model)
            if counts['gate']!=protocol.gate_parameter_count(arm): raise ValueError('Unexpected extra parameter count')
            if arm=='h32_control': common_h32=digest;original_counts=counts
            elif arm.startswith('h32_'):
                if digest!=common_h32: raise ValueError('Three H32 arms did not inherit identical complete source state')
                if counts['bs']!=original_counts['bs'] or counts['full']-original_counts['full']!=counts['gate']:
                    raise ValueError('More than theta changed capacity')
            rows[arm]=dict(source=record,parameters=counts,migration=mapping,common_state_sha256=digest,
                common_construction_sha256=constructed,strict_self_load=True,external_rng_unchanged=True,prior_count=20000)
            if arm=='h32_static_power':
                key=next(iter(source['state']))
                for kind in ('missing','shape','dtype'):
                    bad=dict(source['state'])
                    if kind=='missing':bad.pop(key)
                    elif kind=='shape':bad[key]=bad[key].reshape(-1)[:1]
                    else:bad[key]=bad[key].to(torch.float64)
                    try:migrate_source_state_dict(model,bad)
                    except ValueError:pass
                    else:raise ValueError('Migration accepted incorrect '+kind)
                del bad
            # Ordinary new-stage restore may not silently initialize missing theta.
            if counts['gate']:
                try:model.load_state_dict(source['state'],strict=True)
                except RuntimeError:pass
                else:raise ValueError('Strict new-model load silently accepted missing gate')
            del copied,model,source
        report['initialization'][str(seed)]=dict(arms=rows,three_h32_sources_exactly_paired=True)


def zero_gate(torch,arm,report):
    from train_grouped_hybrid_power import load_source
    from models.grouped_hybrid_anchor import GroupedHybridAnchorPrecoder
    from models.grouped_hybrid_power import GroupedHybridPowerPrecoder,migrate_source_state_dict
    from main_train import SumRateLoss
    seed=SEEDS[0]
    payload,_=load_source(torch,arm,seed)
    old=GroupedHybridAnchorPrecoder(anchor_config(protocol.source_arm(arm),seed))
    old.load_state_dict(payload['state'],strict=True);old.cuda().eval()
    new=GroupedHybridPowerPrecoder(power_config(arm,seed))
    migrate_source_state_dict(new,payload['state']);del payload
    new.cuda().eval()
    generator=torch.Generator(device='cpu').manual_seed(20270104)
    hdl=torch.randn(2,4,32,256,2,generator=generator).contiguous()
    hul=torch.randn(2,4,32,256,2,generator=generator).contiguous()
    hdl=torch.view_as_complex(hdl).cuda();hul=torch.view_as_complex(hul).cuda()
    noise=protocol.standard_noise(seed,'preflight',0,2,'cuda')
    before={k:tensor_digest(dict(model.named_buffers())) for k,model in (('old',old),('new',new))}
    records=report.setdefault('zero_gate',{}).setdefault(arm,{})
    with torch.no_grad():
        for gamma in (0,12.5,25):
            cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
            raw=old.ue(hdl,gamma,old.layout.active_per_user)
            X,_,_=protocol.normalize_and_map(raw,old.layout)
            Y=protocol.transmit(hul,X,old.layout,N0=protocol.noise_variance(gamma),standard_noise=noise)
            old_w,old_h=old.forward_with_representation(hdl,hul,gamma,standard_noise=noise)
            old_loss,_=SumRateLoss()(old_w,hdl,10**(-25/10))
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
            out=new.forward_with_diagnostics(hdl,hul,gamma,standard_noise=noise)
            new_loss,_=SumRateLoss()(out['precoder'],hdl,10**(-25/10))
            comparisons=dict(X=_compare(torch,out['diagnostics']['physical_send'],X),
                Y=_compare(torch,out['diagnostics']['physical_received'],Y),
                H_hat=_compare(torch,out['channel_surrogate'],old_h),
                normalized_RZF=_compare(torch,out['precoder'],old_w),
                sum_rate=_compare(torch,-new_loss,-old_loss))
            records[str(gamma)]=comparisons
            if not all(row['passed'] for row in comparisons.values()):
                raise ValueError('Zero-gate actual-path mismatch: '+arm+' gamma='+str(gamma))
    for key,model in (('old',old),('new',new)):
        if tensor_digest(dict(model.named_buffers()))!=before[key]: raise ValueError('Eval modified buffer '+key)
    del old,new,hdl,hul,noise,out,raw,X,Y,old_w,old_h
    torch.cuda.empty_cache()


def batch48(torch,arm,report):
    from train_grouped_hybrid_power import load_source
    from models.grouped_hybrid_power import GroupedHybridPowerPrecoder,migrate_source_state_dict
    from main_train import SumRateLoss,set_seed
    set_seed(20270104)
    seed=SEEDS[0];payload,_=load_source(torch,arm,seed)
    model=GroupedHybridPowerPrecoder(power_config(arm,seed))
    migrate_source_state_dict(model,payload['state']);del payload
    model.cuda().train()
    row=report.setdefault('batch48',{}).setdefault(arm,dict(passed=False))
    torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    hdl=torch.randn(48,4,32,256,device='cuda',dtype=torch.complex64)
    hul=torch.randn_like(hdl)
    noise=protocol.standard_noise(seed,'preflight',1,48,'cuda')
    before=tensor_digest(model.power_prior.state_dict())
    try:
        w,hat,pending=model.training_with_statistics(hdl,hul,12.5,global_step=20001,standard_noise=noise)
        if before!=tensor_digest(model.power_prior.state_dict()) or pending['read_count']!=20000 or pending['step']!=20001:
            raise ValueError('Forward committed/reset prior or used incorrect cumulative address')
        deployed=model.zf_solver(hat)
        deployed=deployed*torch.sqrt(1/(deployed.abs().square().sum((1,2),keepdim=True)+1e-9))
        loss,_=SumRateLoss()(deployed,hdl,10**(-25/10))
        row.update(power_max_abs=(deployed.abs().square().sum((1,2))-1).abs().max().item(),
            energy_max_abs=(pending['power_mean'].sum(-1)-64).abs().max().item())
        if row['power_max_abs']>POWER_ATOL or row['energy_max_abs']>1e-4 or not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss or invalid energy/power')
        loss.backward()
        active=[(n,p.grad) for n,p in model.named_parameters() if p.grad is not None]
        if not active or any(not torch.isfinite(g).all() for _,g in active): raise FloatingPointError('Nonfinite model gradients')
        theta=model.power_gate.theta
        if theta is not None:
            if theta.grad is None or not torch.isfinite(theta.grad).all() or theta.grad.norm().item()<=0:
                raise ValueError('Zero-initialized gate has no finite nonzero task gradient')
            row['gate_gradient']=theta.grad.detach().cpu().tolist()
        else:row['gate_gradient']=[]
        if any(p.grad is not None for n,p in model.named_parameters() if n in model.inactive_parameter_names()):
            raise ValueError('Stored inactive FDMA UE branch executed')
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
        optimizer.step();model.post_optimizer_commit(pending,20001)
        model.power_prior.validate_state(20001)
        if any(not torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Nonfinite scratch update')
        row.update(prior_before=20000,prior_after=20001,parameters=_parameters(model),
            scratch_only=True,passed=True)
        torch.cuda.synchronize()
    finally:
        row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del model,optimizer,hdl,hul,noise,w,hat,pending,deployed,loss,active
    torch.cuda.empty_cache()


def run(output):
    output=safe_inside(output,OUTPUT_ROOT/'batches');output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,plan=plan())
    originals={}
    try:
        import torch
        from train_grouped_hybrid_power import apply_backend,SOURCE_FILES
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
        apply_backend(torch)
        report.update(backend=_backend_info(torch),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha256(ROOT/p) for p in (*SOURCE_FILES,'preflight_grouped_hybrid_power.py')})
        originals=_source_files();report['source_files_before']=originals
        initialization(torch,report)
        for arm in ARMS:zero_gate(torch,arm,report)
        for arm in ARMS:batch48(torch,arm,report)
        report['passed']=True
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),automatic_fallback=False)
        raise
    finally:
        unchanged=all(Path(p).is_file() and sha256(p)==value for p,value in originals.items())
        report['source_files_unchanged']=bool(originals) and unchanged
        if originals and not unchanged:
            report.update(passed=False,source_integrity_error='Source artifact changed during preflight')
        save_json(output/'report.json',report)
        if originals and not unchanged: raise RuntimeError('Source artifact changed during preflight')
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
