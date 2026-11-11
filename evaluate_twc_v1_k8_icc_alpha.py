#!/usr/bin/env python3
"""Frozen scalar-alpha calibration and held-out six-policy comparison.

No optimizer, backward pass, checkpoint writes or final-data alpha selection.
The serial source/rate models reuse one forward for all relevant alpha values.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

from physical_split_finish_io import write_union_csv
from evaluate_twc_v1_k8_icc_aligned import (
    _tensor_hash, checked_input_hashes, frozen_evaluation,
)


def _runtime():
    import icc_alpha_runtime as runtime
    return runtime


def _config():
    from configs import twc_v1_k8_icc_alpha_config as cfg
    return cfg


def _results():
    import twc_v1_k8_icc_alpha_results as results
    return results


def score_out(out, truth, alpha):
    """Offline physical score; truth never re-enters the neural BS forward."""
    import torch
    import train_twc_v1_k8_icc_aligned as original
    downlink=original.load_local('utils/twc_v1_k8_downlink.py','_icc_alpha_downlink')
    cfg=_config()
    prediction=out['H_hat']
    W,power=downlink.design_rzf(prediction,alpha=alpha)
    rate=downlink.evaluate_sum_rate(W,truth,downlink_snr_db=cfg.DL_SNR)
    error=(power-1.).abs().amax(dim=1)
    if not bool((error<=1e-5).all()):
        raise FloatingPointError('RZF per-tone total power changed')
    vectors=dict(dl_nmse_linear=downlink.downlink_nmse(prediction,truth).mean(dim=1),
        sum_rate_bps_hz=rate['sum_rate_bps_hz'],
        desired_signal_power_mean=rate['desired_signal_power_mean'],
        interference_power_mean=rate['interference_power_mean'],
        min_user_rate_bps_hz=rate['user_rate_bps_hz'].amin(dim=1),
        interference_to_signal_ratio=rate['interference_power_mean']/rate['desired_signal_power_mean'].clamp_min(1e-20),
        predicted_dl_energy=prediction.abs().square().mean(dim=(1,2,3)),
        rzf_power_max_error=error)
    vectors['predicted_to_true_dl_energy_ratio']=vectors['predicted_dl_energy']/truth.abs().square().mean(dim=(1,2,3)).clamp_min(1e-20)
    for u in range(8): vectors[f'user_{u}_rate_bps_hz']=rate['user_rate_bps_hz'][:,u]
    wave=out['waveform']; active=wave.active
    soft=out['soft'][...,:active.shape[-1]]
    data=active.abs().square().sum(dim=-1)
    frame=wave.x_grid.abs().square().sum(dim=-1)
    if not bool(torch.isfinite(data).all() & torch.isclose(data,torch.full_like(data,48.),rtol=1e-5,atol=1e-5).all()):
        raise FloatingPointError('Paid data energy differs from48 per UE')
    if not bool(torch.isfinite(frame).all() & torch.isclose(frame,torch.full_like(frame,64.),rtol=1e-5,atol=1e-5).all()):
        raise FloatingPointError('Paid frame energy differs from64 per UE')
    vectors['data_energy']=data.mean(dim=1); vectors['frame_energy']=frame.mean(dim=1)
    def normalized(a,b):
        return ((a-b).abs().square().sum(-1)/b.abs().square().sum(-1).clamp_min(1e-20)).mean(1)
    vectors['soft_nmse_linear']=normalized(soft,active)
    if active.shape[-1]==44:
        key=active[...,:4].abs().square().sum(-1)
        shared=active[...,4:].abs().square().sum(-1)
        vectors.update(soft_key_nmse_linear=normalized(soft[...,:4],active[...,:4]),
            soft_shared_nmse_linear=normalized(soft[...,4:],active[...,4:]),
            key_energy=key.mean(1),shared_energy=shared.mean(1),
            key_energy_user_min=key.amin(1),key_energy_user_max=key.amax(1),
            shared_energy_user_min=shared.amin(1),shared_energy_user_max=shared.amax(1))
        for u in range(8):
            vectors[f'user_{u}_key_energy']=key[:,u]
            vectors[f'user_{u}_shared_energy']=shared[:,u]
    if 'H_ul_est' in out:
        vectors['estimated_ul_energy']=out['H_ul_est'].abs().square().mean(dim=(1,2,3))
    if any(not bool(torch.isfinite(v).all()) for v in vectors.values()):
        raise FloatingPointError('Nonfinite alpha-evaluation metric')
    return {k:v.detach().cpu().tolist() for k,v in vectors.items()}


def fingerprint_map(entries, *, batches, kinds=('source',)):
    """Require actual channel/noise/pilot equality, not just a shared seed."""
    cfg=_config()
    expected={(kind,arm,bi,float(g)) for kind in kinds for arm in cfg.ARMS
              for bi in range(batches) for g in cfg.GAMMAS}
    result={}
    for row in entries:
        key=(row.get('kind','source'),row['arm'],row['batch_index'],float(row['gamma_db']))
        if key not in expected or key in result:
            raise ValueError('Unexpected or duplicate alpha input fingerprint')
        hashes=row['fingerprint']
        if set(hashes)!={'H_dl','H_ul','standard_noise','pilot_polarity'}:
            raise ValueError('Complete channel/noise/pilot fingerprint required')
        if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
               for v in hashes.values()):
            raise ValueError('Invalid actual input fingerprint')
        result[key]=hashes
    if set(result)!=expected:
        raise ValueError('Incomplete alpha input fingerprint set')
    for bi in range(batches):
        for gamma in cfg.GAMMAS:
            values=[result[kind,arm,bi,float(gamma)] for kind in kinds for arm in cfg.ARMS]
            if any(value!=values[0] for value in values[1:]):
                raise ValueError('Policies/arms used different actual inputs')
    return result


def _setup(output_dir, require_cuda, *, phase):
    cfg,runtime=_config(),_runtime()
    output=runtime.inside(output_dir)
    if phase not in ('calibration','final'):
        raise ValueError('Declared alpha evaluation phase required')
    subtree=(cfg.OUTPUT_ROOT/('calibration' if phase=='calibration' else 'evaluation/final')).resolve()
    if output==subtree or not output.is_relative_to(subtree):
        raise ValueError('Output must be a proper child of its declared phase subtree')
    import torch
    core=runtime.core_identity(); runtime.guard_core(core)
    device=runtime.configure_cuda() if require_cuda else torch.device('cpu')
    output.mkdir(parents=True,exist_ok=False)
    return cfg,runtime,core,device,output


def _validate_vectors(vectors, batch):
    import math
    if not vectors or 'sum_rate_bps_hz' not in vectors:
        raise ValueError('Actual complete-MU rate vector required')
    for name,values in vectors.items():
        if len(values)!=batch or any(isinstance(v,bool) or not isinstance(v,(int,float))
                                    or not math.isfinite(v) for v in values):
            raise ValueError('Invalid complete-MU metric vector: '+name)


def _record_inputs(sample, out, *, kind,arm,bi,gamma,hdl,hul):
    hashes=checked_input_hashes(out.get('input_hashes'),H_dl=hdl,H_ul=hul)
    return dict(kind=kind,arm=arm,batch_index=bi,gamma_db=float(gamma),
                address=sample.get('address'),fingerprint=hashes)


def _sample(data, metadata, phase, bi):
    index=_config().ROLE_RANGES[phase][0]+bi
    sample=data.sample('validation',index)
    if sample.get('address')!=metadata.address('validation',index):
        raise ValueError('Actual alpha data address does not match its declared phase')
    return sample


def _drop_models():
    import torch
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def calibrate(output_dir, *, require_cuda=True, loader=None, data_factory=None,
              forward=None, scorer=None):
    """Select one scalar alpha per arm using calibration addresses only."""
    cfg,runtime,core,device,output=_setup(output_dir,require_cuda,phase='calibration')
    loader=runtime.load_model if loader is None else loader
    data_factory=runtime.make_data if data_factory is None else data_factory
    forward=runtime.forward_codec if forward is None else forward
    scorer=score_out if scorer is None else scorer
    started=time.perf_counter(); rows=[]; fingerprints=[]; records={}; protocol=None
    manifest=dict(format=cfg.NAMESPACE+'_calibration_v1',phase='calibration',status='running',
        core=core,recipe_fingerprint=cfg.fingerprint(),alpha_grid=list(cfg.ALPHA_GRID),
        gamma_db=list(cfg.GAMMAS),batch_size=cfg.MICRO_BATCH,n_batches=cfg.CALIBRATION_BATCHES,
        samples_per_condition=cfg.CALIBRATION_BATCHES*cfg.MICRO_BATCH,
        no_optimizer=True,test_used=False,training_seed_replications=1)
    try:
        for arm in cfg.ARMS:
            codec,stats,ce,record=loader('source',arm,device)
            runtime.verify_files(record); records[arm]=record
            data=data_factory(device)
            metadata=runtime.make_data(device)
            if data.protocol!=metadata.protocol:
                raise ValueError('Calibration data factory changed the declared protocol')
            if protocol is not None and data.protocol!=protocol:
                raise ValueError('Calibration arms changed the data protocol')
            protocol=data.protocol
            with frozen_evaluation({arm:(codec,stats)},ce):
                for bi in range(cfg.CALIBRATION_BATCHES):
                    sample=_sample(data,metadata,'calibration',bi)
                    hdl,hul=_tensor_hash(sample['H_dl']),_tensor_hash(sample['H_ul'])
                    for gi,gamma in enumerate(cfg.GAMMAS):
                        out=forward(codec,stats,ce,sample,gamma_db=gamma,noise_key=('calibration',bi,gi))
                        estimated=_tensor_hash(out['H_hat'])
                        fingerprints.append(_record_inputs(sample,out,kind='source',arm=arm,bi=bi,
                            gamma=gamma,hdl=hdl,hul=hul))
                        for alpha in cfg.ALPHA_GRID:
                            vectors=scorer(out,sample['H_dl'],float(alpha))
                            _validate_vectors(vectors,cfg.MICRO_BATCH)
                            for within,rate in enumerate(vectors['sum_rate_bps_hz']):
                                rows.append(dict(arm=arm,alpha=float(alpha),gamma_db=float(gamma),
                                    sample_index=bi*cfg.MICRO_BATCH+within,batch_index=bi,
                                    within_batch=within,sum_rate_bps_hz=rate))
                        if _tensor_hash(out['H_hat'])!=estimated:
                            raise RuntimeError('Alpha scoring mutated the cached CSI prediction')
                    if _tensor_hash(sample['H_dl'])!=hdl or _tensor_hash(sample['H_ul'])!=hul:
                        raise RuntimeError('Calibration input channels mutated')
            runtime.verify_files(record)
            del codec,stats,ce,data,metadata,out,sample
            _drop_models()
        fingerprint_map(fingerprints,batches=cfg.CALIBRATION_BATCHES)
        selection=_results().select_alphas(rows,samples=cfg.CALIBRATION_BATCHES*cfg.MICRO_BATCH,
                                            batch_size=cfg.MICRO_BATCH)
        write_union_csv(output/'samples.csv.gz',rows)
        write_union_csv(output/'scores.csv',selection['scores'])
        runtime.atomic_json(output/'input_fingerprints.json',fingerprints,exclusive=True)
        candidate=dict(selection,core=core,data_protocol=protocol,recipe_fingerprint=cfg.fingerprint())
        runtime.atomic_json(output/'selection_candidate.json',candidate,exclusive=True)
        files={name:runtime.sha256(output/name) for name in (
            'samples.csv.gz','scores.csv','input_fingerprints.json','selection_candidate.json')}
        runtime.guard_core(core)
        manifest.update(status='passed',data_protocol=protocol,source_records=records,files=files,
            elapsed_seconds=time.perf_counter()-started,frozen_models_verified=True,
            alpha_selection_uses_final_data=False)
        runtime.atomic_json(output/'manifest.json',manifest,exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if not (output/'manifest.json').exists():
            runtime.atomic_json(output/'manifest.json',manifest,exclusive=True)
        raise


def evaluate_final(output_dir, selection_path, *, require_cuda=True, loader=None,
                   data_factory=None, forward=None, scorer=None):
    """Score all three policies only after a validated calibration lock exists."""
    cfg,runtime=_config(),_runtime()
    selection=runtime.load_selection(selection_path)
    core=runtime.core_identity()
    if selection.get('core')!=core or set(selection.get('selected',{}))!=set(cfg.ARMS):
        raise ValueError('Selection lock identity/arms changed')
    if any(isinstance(a,bool) or a not in cfg.ALPHA_GRID for a in selection['selected'].values()):
        raise ValueError('Locked alpha is outside the declared grid')
    cfg,runtime,core,device,output=_setup(output_dir,require_cuda,phase='final')
    loader=runtime.load_model if loader is None else loader
    data_factory=runtime.make_data if data_factory is None else data_factory
    forward=runtime.forward_codec if forward is None else forward
    scorer=score_out if scorer is None else scorer
    started=time.perf_counter();rows=[];fingerprints=[];records={};protocol=None
    policy_alphas={cfg.method_name(policy,arm):float(selection['selected'][arm] if policy=='source_calibrated'
        else cfg.FIXED_ALPHA) for policy in cfg.POLICIES for arm in cfg.ARMS}
    manifest=dict(format=cfg.NAMESPACE+'_final_v1',phase='final',status='running',core=core,
        recipe_fingerprint=cfg.fingerprint(),selection_record_id=selection['record_id'],
        gamma_db=list(cfg.GAMMAS),batch_size=cfg.MICRO_BATCH,n_batches=cfg.FINAL_BATCHES,
        samples_per_condition=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,methods=list(cfg.METHODS),
        policy_alphas=policy_alphas,no_optimizer=True,test_used=False,training_seed_replications=1)
    try:
        for kind in ('source','rate'):
            policies=('source_fixed','source_calibrated') if kind=='source' else ('rate_fixed',)
            for arm in cfg.ARMS:
                codec,stats,ce,record=loader(kind,arm,device)
                runtime.verify_files(record);records[kind+'_'+arm]=record
                data=data_factory(device)
                metadata=runtime.make_data(device)
                if data.protocol!=metadata.protocol:
                    raise ValueError('Final data factory changed the declared protocol')
                if protocol is not None and data.protocol!=protocol:
                    raise ValueError('Final models changed the data protocol')
                protocol=data.protocol
                if protocol!=selection['data_protocol']:
                    raise ValueError('Calibration/final data protocol changed')
                with frozen_evaluation({arm:(codec,stats)},ce):
                    for bi in range(cfg.FINAL_BATCHES):
                        sample=_sample(data,metadata,'final',bi)
                        hdl,hul=_tensor_hash(sample['H_dl']),_tensor_hash(sample['H_ul'])
                        for gi,gamma in enumerate(cfg.GAMMAS):
                            out=forward(codec,stats,ce,sample,gamma_db=gamma,noise_key=('final',bi,gi))
                            estimate=_tensor_hash(out['H_hat'])
                            fingerprints.append(_record_inputs(sample,out,kind=kind,arm=arm,bi=bi,
                                gamma=gamma,hdl=hdl,hul=hul))
                            for policy in policies:
                                method=cfg.method_name(policy,arm);alpha=policy_alphas[method]
                                vectors=scorer(out,sample['H_dl'],alpha)
                                _validate_vectors(vectors,cfg.MICRO_BATCH)
                                for within in range(cfg.MICRO_BATCH):
                                    rows.append(dict(arm=method,gamma_db=float(gamma),
                                        sample_index=bi*cfg.MICRO_BATCH+within,batch_index=bi,
                                        within_batch=within,rzf_alpha=alpha,
                                        **{name:value[within] for name,value in vectors.items()}))
                            if _tensor_hash(out['H_hat'])!=estimate:
                                raise RuntimeError('Policy scoring mutated the cached CSI prediction')
                        if _tensor_hash(sample['H_dl'])!=hdl or _tensor_hash(sample['H_ul'])!=hul:
                            raise RuntimeError('Final input channels mutated')
                runtime.verify_files(record)
                del codec,stats,ce,data,metadata,out,sample
                _drop_models()
        fingerprint_map(fingerprints,batches=cfg.FINAL_BATCHES,kinds=('source','rate'))
        summary=_results().summarize_rows(rows,samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
                                         batch_size=cfg.MICRO_BATCH)
        write_union_csv(output/'samples.csv.gz',rows)
        write_union_csv(output/'summary.csv',summary)
        comparisons=_results().paired_comparisons(rows,samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
                                                  batch_size=cfg.MICRO_BATCH)
        write_union_csv(output/'comparisons.csv',comparisons)
        runtime.atomic_json(output/'input_fingerprints.json',fingerprints,exclusive=True)
        files={name:runtime.sha256(output/name) for name in (
            'samples.csv.gz','summary.csv','comparisons.csv','input_fingerprints.json')}
        runtime.guard_core(core)
        # Re-read the immutable lock after all inference, including its original
        # calibration artifacts. Final scoring never updates its selected alpha.
        if runtime.load_selection(selection_path)!=selection:
            raise ValueError('Selection lock changed during final scoring')
        manifest.update(status='passed',data_protocol=protocol,source_records=records,files=files,
            elapsed_seconds=time.perf_counter()-started,frozen_models_verified=True,
            outputs=dict(summary=str(output/'summary.csv'),comparisons=str(output/'comparisons.csv')))
        runtime.atomic_json(output/'manifest.json',manifest,exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if not (output/'manifest.json').exists():
            runtime.atomic_json(output/'manifest.json',manifest,exclusive=True)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--run',action='store_true');actions.add_argument('--dry-run',action='store_true')
    parser.add_argument('--phase',choices=('calibration','final'),default='final')
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--selection',type=Path)
    args=parser.parse_args(argv);cfg=_config()
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE,phase=args.phase,alpha_grid=cfg.ALPHA_GRID,
            calibration_MU=cfg.CALIBRATION_BATCHES*cfg.MICRO_BATCH,
            final_MU=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,global_alpha_per_arm=True,
            optimizer_updates=0,test_used=False,writes=False,weights_read=False),indent=2));return 0
    if not args.run:parser.print_help();return 0
    if args.output_dir is None:parser.error('--run requires --output-dir')
    if args.phase=='calibration':
        calibrate(args.output_dir)
    else:
        if args.selection is None:parser.error('Final scoring requires --selection')
        evaluate_final(args.output_dir,args.selection)
    return 0


if __name__=='__main__':raise SystemExit(main())
