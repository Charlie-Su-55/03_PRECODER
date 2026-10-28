#!/usr/bin/env python3
"""F-only CUDA acceptance; no historical-model forward or RNG-suite rerun."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import traceback

from as_receiver_batch_support import safe_inside, sha256
from preflight_proposal_as_direct import save_report, _backend_info
from preflight_joint_swin import tensor_digest
from preflight_grouped_codec import required_modules, _receiver_boundary, POWER_ATOL

ROOT = Path(__file__).resolve().parent
EXPECTED_PARAMETERS = dict(ue=10877090, bs=34933678, full=45810768)


def plan():
    return dict(arm='legacy_fdma', batch_size=48, seeds=[20261030,20261031],
        static_parameter_expectation=EXPECTED_PARAMETERS, counts_measured=False,
        active_slots_per_subband=16, active_per_ue=64, total_occupied_re=256,
        encoder_branch='original AS, not original FDMA', energy_per_ue=64,
        cuda_required=True, synthetic_channels=True, historical_model_forwards=False,
        disposable_optimizer_update='one scratch update for memory/finite checks; no saved weights or official updates',
        no_automatic_fallback=True)


def _same_tensors(torch, expected, actual, label):
    if set(expected) != set(actual):
        raise ValueError(label+' keys differ')
    for name, tensor in expected.items():
        other = actual[name]
        if tensor.shape != other.shape or tensor.dtype != other.dtype or not torch.equal(tensor, other):
            raise ValueError(label+' tensor differs: '+name)


def initialization(torch, old, report):
    from configs.grouped_codec_fdma_train_config import fdma_config, SEEDS, protocol
    from configs.grouped_codec_train_config import grouped_codec_config
    from models.grouped_codec_fdma import GroupedCodecFDMAPrecoder
    from models.grouped_codec_precoder import LegacyUE, GroupedJointSwinDecoder
    from train_grouped_codec_fdma import validate_model_identity, validate_model_state
    archived = old['summary']['preflight']['initialization']
    report['initialization'] = {}
    for seed in SEEDS:
        caller_rng = torch.get_rng_state().clone()
        model = GroupedCodecFDMAPrecoder(fdma_config(seed))
        validate_model_identity(fdma_config(seed), model)
        validate_model_state(torch, model, model.state_dict())
        model.load_state_dict(model.state_dict(), strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(protocol.RandomAddress(seed,'init',0,'ue_legacy').seed_value)
            reference_ue = LegacyUE(grouped_codec_config('legacy_all4',seed))
        _same_tensors(torch, dict(reference_ue.named_parameters()), dict(model.ue.named_parameters()), 'Legacy UE')
        del reference_ue
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(protocol.RandomAddress(seed,'init',0,'bs').seed_value)
            reference_bs = GroupedJointSwinDecoder(protocol.LayoutSpec('fdma'))
        _same_tensors(torch, dict(reference_bs.named_parameters()), dict(model.decoder.named_parameters()), 'Common BS')
        _same_tensors(torch, dict(reference_bs.named_buffers()), dict(model.decoder.named_buffers()), 'E-layout buffers')
        del reference_bs
        if not torch.equal(caller_rng, torch.get_rng_state()):
            raise RuntimeError('F/reference construction changed caller CPU RNG')
        ue_hash=tensor_digest({n:p for n,p in model.named_parameters() if n.startswith('ue.')})
        bs_hash=tensor_digest({n:p for n,p in model.named_parameters() if n.startswith('decoder.')})
        previous=archived[str(seed)]
        if any(previous[name]['ue_sha256']!=ue_hash for name in ('legacy_all4','legacy_group2')):
            raise ValueError('F UE differs from archived A/C initialization')
        if any(value['bs_sha256']!=bs_hash for value in previous.values()):
            raise ValueError('F BS differs from archived A-E initialization')
        report['initialization'][str(seed)] = dict(ue_sha256=ue_hash,bs_sha256=bs_hash,
            per_tensor_matches=True, archived_initialization_matches=True,
            fdma_layout_buffers_equal=True, strict_self_load=True, caller_rng_unchanged=True)
        del model


def batch48(torch, report):
    from configs.grouped_codec_fdma_train_config import fdma_config, protocol
    from models.grouped_codec_fdma import GroupedCodecFDMAPrecoder
    from main_train import set_seed, SumRateLoss
    cfg=fdma_config(20261030)
    set_seed(cfg.SEED)
    model=GroupedCodecFDMAPrecoder(cfg).cuda().train()
    counts=dict(ue=sum(p.numel() for p in model.ue_parameters()),
                bs=sum(p.numel() for p in model.bs_parameters()),full=sum(p.numel() for p in model.parameters()))
    if counts!=EXPECTED_PARAMETERS:
        raise RuntimeError(f'Actual parameter count differs: {counts}')
    torch.cuda.reset_peak_memory_stats()
    result=report['batch48']=dict(parameters=counts,passed=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    hdl=torch.randn(48,4,32,256,device='cuda',dtype=torch.complex64)
    hul=torch.randn_like(hdl)
    eps=protocol.standard_noise(cfg.SEED,'preflight',0,48,'cuda')
    modules=dict(model.named_modules())
    names=required_modules()+['ue.encoder.aircomp_backbone','ue.condition','ue.encoder.fdma_backbone']
    calls={name:0 for name in names}
    handles=[]
    def counter(name):
        def hook(*unused):calls[name]+=1
        return hook
    try:
        for name in names:handles.append(modules[name].register_forward_hook(counter(name)))
        output=model.forward_with_diagnostics(hdl,hul,10.,standard_noise=eps)
        result['execution_counts']=dict(calls)
        if calls['ue.encoder.fdma_backbone']!=0 or any(calls[n]<1 for n in names if n!='ue.encoder.fdma_backbone'):
            raise RuntimeError('F must execute AS UE and the complete Joint BS, never old FDMA encoder')
        d=output['diagnostics']
        active=d['active_mask']
        if not torch.equal(active,protocol.tensor_geometry(protocol.LayoutSpec('fdma'),'cuda')['active_mask']):
            raise RuntimeError('Incorrect canonical FDMA mask')
        if bool((d['true_z'][:,~active]!=0).any()) or bool((d['z_corrected'][:,~active]!=0).any()):
            raise RuntimeError('Inactive canonical slots nonzero')
        if not torch.allclose(d['energy'],torch.full_like(d['energy'],64),rtol=0,atol=1e-4):
            raise RuntimeError('F changed per-UE total energy')
        empirical=(eps*math.sqrt(protocol.noise_variance(10)/2)).abs().square().mean((1,2))
        if not torch.equal(empirical,d['noise_power']) or d['N0']!=protocol.noise_variance(10):
            raise RuntimeError('Absolute noise convention changed')
        hat=output['channel_surrogate']
        if tuple(hat.shape)!=(48,4,32,256):raise RuntimeError('Incomplete channel representation')
        w=model.zf_solver(hat)
        w=w*torch.sqrt(1/(w.abs().square().sum((1,2),keepdim=True)+1e-9))
        loss,rate=SumRateLoss()(w,hdl,10**(-25/10))
        result.update(power_max_abs=(w.abs().square().sum((1,2))-1).abs().max().item(),
            energy_max_abs=(d['energy']-64).abs().max().item(),N0=d['N0'],synthetic_rate_not_result=rate)
        if not torch.isfinite(loss).all() or result['power_max_abs']>POWER_ATOL:
            raise RuntimeError('Nonfinite loss or RZF power violation')
        loss.backward()
        result['gradients']={}
        for name in names:
            grads=[p.grad for p in modules[name].parameters() if p.grad is not None]
            norm=sum(g.detach().double().square().sum().item() for g in grads)**.5
            finite=all(torch.isfinite(g).all().item() for g in grads)
            result['gradients'][name]=dict(tensors=len(grads),norm=norm,finite=finite)
            if name=='ue.encoder.fdma_backbone':
                if grads:raise RuntimeError('Inactive legacy FDMA branch received gradients')
            elif not grads or not finite or norm<=0:raise RuntimeError('Broken active gradient: '+name)
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('Nonfinite scratch update')
        torch.cuda.synchronize()
    finally:
        for handle in handles:handle.remove()
        result.update(hooks_removed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    del output,hat,w,loss,d,optimizer,hdl,hul,eps
    model.zero_grad(set_to_none=True)
    result['inactive_boundary']=_receiver_boundary(torch,model)
    result['passed']=True


def run(output,old_evidence):
    from configs.grouped_codec_fdma_train_config import OUTPUT_ROOT
    output=safe_inside(output,OUTPUT_ROOT/'batches');output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,plan=plan())
    try:
        import torch
        from train_grouped_codec_fdma import apply_backend, SOURCE_FILES
        from evaluate_grouped_codec_fdma import verify_old_batch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        old=verify_old_batch()
        expected=json.loads(Path(old_evidence).read_text(encoding='utf-8'))
        if old['frozen_files']!=expected['frozen_files']:raise RuntimeError('Old evidence changed since controller check')
        apply_backend(torch)
        report.update(backend=_backend_info(torch),old_summary_sha256=old['summary_sha256'],
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha256(ROOT/p) for p in (*SOURCE_FILES,'preflight_grouped_codec_fdma.py')})
        initialization(torch,old,report)
        batch48(torch,report)
        report['passed']=True
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise
    finally:
        save_report(output,report)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--old-evidence',type=Path)
    args=parser.parse_args(argv)
    if args.run:
        if args.output_dir is None or args.old_evidence is None:parser.error('--run requires output and old evidence')
        run(args.output_dir,args.old_evidence)
    elif args.output_dir is not None or args.old_evidence is not None:parser.error('Paths require --run')
    elif args.dry_run:print(json.dumps(plan(),indent=2))
    else:parser.print_help()
    return 0


if __name__=='__main__':raise SystemExit(main())
