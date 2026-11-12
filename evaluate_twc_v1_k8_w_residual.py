#!/usr/bin/env python3
"""Frozen direct-precoder evaluation of the six residual-W policy/arm methods.

True downlink channels enter only the offline scorer. The transmitted precoder
comes from the BS forward; this evaluator never replaces it with a new RZF.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import traceback

from physical_split_finish_io import write_union_csv
from evaluate_twc_v1_k8_icc_aligned import (
    _tensor_hash, checked_input_hashes, frozen_evaluation,
)


def _config():
    from configs import twc_v1_k8_w_residual_config as cfg
    return cfg


def _runtime():
    import w_residual_runtime as runtime
    return runtime


def _results():
    import twc_v1_k8_w_residual_results as results
    return results


def score_out(out, truth):
    """Score the actual returned W, with truth used only for these metrics."""
    import torch
    from train_twc_v1_k8_icc_aligned import load_local
    downlink=load_local('utils/twc_v1_k8_downlink.py','_w_residual_downlink')
    cfg=_config(); prediction=out['H_hat']
    W,base,delta=(out[key] for key in ('W','W_base','delta_W'))
    for name,value in (('W',W),('W_base',base),('delta_W',delta)):
        if (not torch.is_tensor(value) or value.dtype!=torch.complex64 or value.ndim!=4
                or tuple(value.shape)!=(truth.shape[0],256,32,8)
                or not bool(torch.isfinite(value).all())):
            raise ValueError(name+' must be finite complex64 [B,256,32,8]')
    power=W.abs().square().sum(dim=(-2,-1))
    base_power=base.abs().square().sum(dim=(-2,-1))
    error=(power-1.).abs().amax(dim=1)
    base_error=(base_power-1.).abs().amax(dim=1)
    if not bool((error<=1e-5).all() & (base_error<=1e-5).all()):
        raise FloatingPointError('Actual/base precoder per-tone total power differs from one')
    rate=downlink.evaluate_sum_rate(W,truth,downlink_snr_db=cfg.DL_SNR)
    delta_norm=delta.abs().square().sum(dim=(-2,-1)).sqrt()
    effective_norm=(W-base).abs().square().sum(dim=(-2,-1)).sqrt()
    base_norm=base_power.sqrt().clamp_min(1e-20)
    vectors=dict(dl_nmse_linear=downlink.downlink_nmse(prediction,truth).mean(dim=1),
        sum_rate_bps_hz=rate['sum_rate_bps_hz'],
        desired_signal_power_mean=rate['desired_signal_power_mean'],
        interference_power_mean=rate['interference_power_mean'],
        min_user_rate_bps_hz=rate['user_rate_bps_hz'].amin(dim=1),
        interference_to_signal_ratio=rate['interference_power_mean']/rate['desired_signal_power_mean'].clamp_min(1e-20),
        predicted_dl_energy=prediction.abs().square().mean(dim=(1,2,3)),
        precoder_power_max_error=error,base_precoder_power_max_error=base_error,
        rzf_power_max_error=base_error,
        delta_w_norm=delta_norm.mean(1),delta_w_relative_norm=(delta_norm/base_norm).mean(1),
        effective_delta_w_norm=effective_norm.mean(1),
        effective_delta_w_relative_norm=(effective_norm/base_norm).mean(1))
    vectors['predicted_to_true_dl_energy_ratio']=vectors['predicted_dl_energy']/truth.abs().square().mean(dim=(1,2,3)).clamp_min(1e-20)
    for user in range(8):
        vectors[f'user_{user}_rate_bps_hz']=rate['user_rate_bps_hz'][:,user]
    wave=out['waveform'];active=wave.active;soft=out['soft'][...,:active.shape[-1]]
    data=active.abs().square().sum(-1);frame=wave.x_grid.abs().square().sum(-1)
    if not bool(torch.isfinite(data).all() & torch.isclose(data,torch.full_like(data,48.),rtol=1e-5,atol=1e-5).all()):
        raise FloatingPointError('Paid data energy differs from48 per UE')
    if not bool(torch.isfinite(frame).all() & torch.isclose(frame,torch.full_like(frame,64.),rtol=1e-5,atol=1e-5).all()):
        raise FloatingPointError('Paid frame energy differs from64 per UE')
    vectors['data_energy']=data.mean(1);vectors['frame_energy']=frame.mean(1)
    def normalized(a,b):
        return ((a-b).abs().square().sum(-1)/b.abs().square().sum(-1).clamp_min(1e-20)).mean(1)
    vectors['soft_nmse_linear']=normalized(soft,active)
    if active.shape[-1]==44:
        key=active[...,:4].abs().square().sum(-1);shared=active[...,4:].abs().square().sum(-1)
        vectors.update(soft_key_nmse_linear=normalized(soft[...,:4],active[...,:4]),
            soft_shared_nmse_linear=normalized(soft[...,4:],active[...,4:]),
            key_energy=key.mean(1),shared_energy=shared.mean(1),
            key_energy_user_min=key.amin(1),key_energy_user_max=key.amax(1),
            shared_energy_user_min=shared.amin(1),shared_energy_user_max=shared.amax(1))
        for user in range(8):
            vectors[f'user_{user}_key_energy']=key[:,user]
            vectors[f'user_{user}_shared_energy']=shared[:,user]
    if 'H_ul_est' in out:
        vectors['estimated_ul_energy']=out['H_ul_est'].abs().square().mean(dim=(1,2,3))
    if any(not bool(torch.isfinite(vector).all()) for vector in vectors.values()):
        raise FloatingPointError('Nonfinite residual-W evaluation metric')
    return {key:value.detach().cpu().tolist() for key,value in vectors.items()}


def fingerprint_map(entries, *, batches):
    """Require byte-identical actual channels, standard noise and pilots."""
    cfg=_config()
    expected={(method,bi,float(gamma)) for method in cfg.METHODS
              for bi in range(batches) for gamma in cfg.GAMMAS}
    found={}
    for row in entries:
        key=(row['method'],row['batch_index'],float(row['gamma_db']))
        if key not in expected or key in found:
            raise ValueError('Unexpected or duplicate residual-W input fingerprint')
        hashes=row['fingerprint']
        if set(hashes)!={'H_dl','H_ul','standard_noise','pilot_polarity'}:
            raise ValueError('Complete channel/noise/pilot fingerprint required')
        if any(not isinstance(value,str) or len(value)!=64
               or any(c not in '0123456789abcdef' for c in value) for value in hashes.values()):
            raise ValueError('Invalid actual input fingerprint')
        found[key]=hashes
    if set(found)!=expected:
        raise ValueError('Incomplete residual-W input fingerprint set')
    for bi in range(batches):
        for gamma in cfg.GAMMAS:
            values=[found[method,bi,float(gamma)] for method in cfg.METHODS]
            if any(value!=values[0] for value in values[1:]):
                raise ValueError('Policies/arms used different actual inputs')
    return found


def _checked_output(output_dir,phase):
    cfg,runtime=_config(),_runtime()
    output=runtime.inside(output_dir)
    if phase not in ('preflight','final'):
        raise ValueError('Declared residual-W evaluation phase required')
    subtree=(cfg.OUTPUT_ROOT/('preflight' if phase=='preflight' else 'evaluation/final')).resolve()
    if output==subtree or not output.is_relative_to(subtree):
        raise ValueError('Output must be a proper child of its declared phase subtree')
    return output


def _setup(output_dir,require_cuda):
    cfg,runtime=_config(),_runtime()
    output=_checked_output(output_dir,'final')
    import torch
    core=runtime.core_identity();runtime.guard_core(core)
    device=runtime.configure_cuda() if require_cuda else torch.device('cpu')
    output.mkdir(parents=True,exist_ok=False)
    return cfg,runtime,core,device,output


def _validate_vectors(vectors,batch):
    import math
    if not vectors or 'sum_rate_bps_hz' not in vectors:
        raise ValueError('Actual complete-MU rate vector required')
    for name,values in vectors.items():
        if len(values)!=batch or any(isinstance(value,bool) or not isinstance(value,(int,float))
                                    or not math.isfinite(value) for value in values):
            raise ValueError('Invalid complete-MU metric vector: '+name)


def _sample(data,metadata,phase,bi):
    first,last=_config().ROLE_RANGES[phase]
    if type(bi) is not int or not 0<=bi<=last-first:
        raise ValueError('Sample lies outside the declared phase range')
    index=first+bi;sample=data.sample('validation',index)
    if sample.get('address')!=metadata.address('validation',index):
        raise ValueError('Actual residual-W data address does not match its declared phase')
    return sample


def _drop_models():
    import torch
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def evaluate_final(output_dir,*,require_cuda=True,loader=None,data_factory=None,
                   forward=None,scorer=None):
    """Evaluate each frozen policy serially on the same new final inputs."""
    # Production evaluation must establish actual training-input pairing before
    # creating final artifacts or loading any model. Injected CPU loaders have
    # an explicit fixture provenance instead of claiming real training evidence.
    _checked_output(output_dir,'final')
    production_loader=loader is None
    if production_loader:
        pairing=_runtime().training_pairing_evidence()
        if pairing.get('status')!='passed':
            raise ValueError('Actual saved training-input pairing evidence required')
    else:
        pairing=dict(status='test_fixture',scope='injected evaluation fixture; production training inputs not checked')
    cfg,runtime,core,device,output=_setup(output_dir,require_cuda)
    loader=runtime.load_model if loader is None else loader
    data_factory=runtime.make_data if data_factory is None else data_factory
    forward=runtime.forward_codec if forward is None else forward
    scorer=score_out if scorer is None else scorer
    started=time.perf_counter();rows=[];fingerprints=[];records={};protocol=None
    manifest=dict(format=cfg.NAMESPACE+'_final_v1',phase='final',status='running',core=core,
        recipe_fingerprint=cfg.study_fingerprint(),gamma_db=list(cfg.GAMMAS),
        batch_size=cfg.MICRO_BATCH,n_batches=cfg.FINAL_BATCHES,
        samples_per_condition=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,methods=list(cfg.METHODS),
        no_optimizer=True,test_used=False,training_seed_replications=1,
        training_pairing_verified=production_loader,training_pairing_scope=pairing['scope'],
        delta_w_definition='head residual before final per-tone normalization',
        effective_delta_w_definition='actual transmitted W minus normalized W_base',
        rzf_power_max_error_definition='base RZF power error; actual W uses precoder_power_max_error')
    try:
        runtime.atomic_json(output/'training_pairing.json',pairing,exclusive=True)
        for policy in cfg.POLICIES:
            for arm in cfg.ARMS:
                method=cfg.method_name(policy,arm)
                codec,stats,ce,record=loader(policy,arm,device)
                runtime.verify_files(record);records[method]=record
                data=data_factory(device);metadata=runtime.make_data(device)
                if data.protocol!=metadata.protocol:
                    raise ValueError('Final data factory changed the declared protocol')
                if protocol is not None and data.protocol!=protocol:
                    raise ValueError('Final methods changed the data protocol')
                protocol=data.protocol
                with frozen_evaluation({method:(codec,stats)},ce):
                    for bi in range(cfg.FINAL_BATCHES):
                        sample=_sample(data,metadata,'final',bi)
                        hdl,hul=_tensor_hash(sample['H_dl']),_tensor_hash(sample['H_ul'])
                        for gi,gamma in enumerate(cfg.GAMMAS):
                            out=forward(codec,stats,ce,sample,gamma_db=gamma,noise_key=('final',bi,gi))
                            before={name:_tensor_hash(out[name]) for name in ('H_hat','W','W_base','delta_W')}
                            fingerprints.append(dict(method=method,policy=policy,arm=arm,
                                batch_index=bi,gamma_db=float(gamma),address=sample['address'],
                                fingerprint=checked_input_hashes(out.get('input_hashes'),H_dl=hdl,H_ul=hul)))
                            vectors=scorer(out,sample['H_dl']);_validate_vectors(vectors,cfg.MICRO_BATCH)
                            for within in range(cfg.MICRO_BATCH):
                                rows.append(dict(arm=method,gamma_db=float(gamma),
                                    sample_index=bi*cfg.MICRO_BATCH+within,batch_index=bi,within_batch=within,
                                    **{name:values[within] for name,values in vectors.items()}))
                            if any(_tensor_hash(out[name])!=digest for name,digest in before.items()):
                                raise RuntimeError('Offline scoring mutated a cached BS output')
                        if _tensor_hash(sample['H_dl'])!=hdl or _tensor_hash(sample['H_ul'])!=hul:
                            raise RuntimeError('Final input channels mutated')
                runtime.verify_files(record)
                del codec,stats,ce,data,metadata,out,sample
                _drop_models()
        fingerprint_map(fingerprints,batches=cfg.FINAL_BATCHES)
        summary=_results().summarize_rows(rows,samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,batch_size=cfg.MICRO_BATCH)
        comparisons=_results().paired_comparisons(rows,samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,batch_size=cfg.MICRO_BATCH)
        write_union_csv(output/'samples.csv.gz',rows);write_union_csv(output/'summary.csv',summary)
        write_union_csv(output/'comparisons.csv',comparisons)
        runtime.atomic_json(output/'input_fingerprints.json',fingerprints,exclusive=True)
        files={name:runtime.sha256(output/name) for name in (
            'samples.csv.gz','summary.csv','comparisons.csv','input_fingerprints.json','training_pairing.json')}
        runtime.guard_core(core)
        manifest.update(status='passed',data_protocol=protocol,source_records=records,files=files,
            elapsed_seconds=time.perf_counter()-started,frozen_models_verified=True,
            transmitted_precoder_scored=True,outputs=dict(summary=str(output/'summary.csv'),
                                                         comparisons=str(output/'comparisons.csv')))
        runtime.atomic_json(output/'manifest.json',manifest,exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if not (output/'manifest.json').exists():runtime.atomic_json(output/'manifest.json',manifest,exclusive=True)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--run',action='store_true');actions.add_argument('--dry-run',action='store_true')
    parser.add_argument('--phase',choices=('preflight','final'),default='final')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv);cfg=_config()
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE,phase=args.phase,methods=cfg.METHODS,
            gamma_db=cfg.GAMMAS,final_MU=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
            optimizer_updates=0,transmitted_precoder_scored=True,test_used=False,
            writes=False,weights_read=False),indent=2));return 0
    if not args.run:parser.print_help();return 0
    if args.output_dir is None:parser.error('--run requires --output-dir')
    output=_checked_output(args.output_dir,args.phase)
    if args.phase=='preflight':_runtime().preflight_sources(output,require_cuda=True)
    else:evaluate_final(output)
    return 0


if __name__=='__main__':raise SystemExit(main())
