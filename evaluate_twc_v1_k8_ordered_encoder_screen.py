#!/usr/bin/env python3
"""Paired validation of the four final ordered-window encoder screen models."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

from configs import twc_v1_k8_ordered_encoder_screen_config as cfg

METRICS = ('dl_nmse','sum_rate_bps_hz','desired_signal_power','interference_power')
DIAGNOSTICS = ('data_energy_per_ue','frame_energy_per_ue','rzf_power_mean')
POLICIES = {'hybrid_1_3': (1,3), 'hybrid_1_1': (1,1)}
ARTIFACTS = ('summary.csv','samples.csv.gz','paired_deltas.csv','parameters.csv','results.json')


def conditions():
    return tuple((arm,1 if cfg.is_fdma(arm) else 4,policy,gamma)
        for arm in cfg.ARMS for policy in (('fdma',) if cfg.is_fdma(arm) else tuple(POLICIES))
        for gamma in cfg.GAMMAS)


def statistics_for(values):
    if not values or any(isinstance(v,bool) or not math.isfinite(v) for v in values):
        raise ValueError('Finite nonempty signed values required')
    mean=statistics.fmean(values)
    se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None
    return dict(mean=mean,se=se,ci95_low=None if se is None else mean-1.96*se,
                ci95_high=None if se is None else mean+1.96*se)


def summarize_samples(samples,count=256,batch_size=4):
    if type(count) is not int or count<batch_size or count%batch_size or batch_size!=4:
        raise ValueError('Complete batches of four multi-user channel samples required')
    indexed={condition:{} for condition in conditions()};fingerprints={}
    for row in samples:
        key=tuple(row.get(k) for k in ('arm','group_count','energy_policy','gamma_db'))
        i=row.get('sample_index')
        if key not in indexed or type(i) is not int or not 0<=i<count or i in indexed[key]:
            raise ValueError('Unexpected or duplicate complete ordered-screen sample key')
        if (row.get('seed')!=cfg.SEED or row.get('checkpoint_step')!=cfg.UPDATES
                or row.get('cumulative_updates')!=cfg.SOURCE_UPDATES+cfg.UPDATES
                or row.get('batch_index')!=i//batch_size or row.get('pilot_energy_per_ue')!=16.
                or row.get('downlink_snr_db')!=25.):raise ValueError('Sample protocol/update identity changed')
        fingerprint=row.get('channel_fingerprint')
        if (not isinstance(fingerprint,str) or len(fingerprint)!=64
                or any(c not in '0123456789abcdef' for c in fingerprint)):
            raise ValueError('Actual common channel fingerprint required')
        if fingerprints.setdefault(i//batch_size,fingerprint)!=fingerprint:
            raise ValueError('Methods/conditions do not share actual channels')
        for metric in METRICS+DIAGNOSTICS:
            value=row.get(metric)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Nonfinite/negative metric: '+metric)
        for metric,tolerance in (('energy_max_abs_error',1e-4),('rzf_power_max_abs_error',1e-5)):
            value=row.get(metric)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=tolerance:
                raise ValueError('Energy/RZF constraint failed: '+metric)
        if (abs(row['data_energy_per_ue']-48.)>1e-4 or abs(row['frame_energy_per_ue']-64.)>1e-4
                or abs(row['rzf_power_mean']-1.)>1e-5):raise ValueError('Energy or RZF power units changed')
        indexed[key][i]=row
    summary=[]
    for key,records in indexed.items():
        if set(records)!=set(range(count)):raise ValueError('Incomplete arm/energy/gamma coverage')
        row=dict(zip(('arm','group_count','energy_policy','gamma_db'),key))
        row.update(seed=cfg.SEED,num_samples=count,num_batches=count//batch_size,batch_size=batch_size,
            checkpoint_step=cfg.UPDATES,cumulative_updates=cfg.SOURCE_UPDATES+cfg.UPDATES,
            source_commit=cfg.SOURCE_COMMIT,role='validation',test_used=False)
        for metric in METRICS+DIAGNOSTICS:
            row.update({metric+'_'+name:value for name,value in
                statistics_for([records[i][metric] for i in range(count)]).items()})
        row['energy_max_abs_error']=max(r['energy_max_abs_error'] for r in records.values())
        row['rzf_power_max_abs_error']=max(r['rzf_power_max_abs_error'] for r in records.values())
        summary.append(row)
    return summary,indexed


def contrasts():
    """Coefficients act on raw paired samples, including the interaction."""
    result=[dict(name='ordered_fdma_minus_mean_fdma',policy='fdma',coefficients={
        ('ordered_fdma',1,'fdma'):1,('mean_fdma_control',1,'fdma'):-1})]
    for policy in POLICIES:
        og=('ordered_g4',4,policy);mg=('mean_g4_control',4,policy)
        of=('ordered_fdma',1,'fdma');mf=('mean_fdma_control',1,'fdma')
        for name,coefficients in (
            ('ordered_g4_minus_mean_g4',{og:1,mg:-1}),
            ('ordered_g4_minus_ordered_fdma',{og:1,of:-1}),
            ('gap_change',{og:1,of:-1,mg:-1,mf:1}),
            ('mean_g4_minus_mean_fdma',{mg:1,mf:-1})):
            result.append(dict(name=name+'_'+policy,policy=policy,coefficients=coefficients))
    return result


def paired_statistics(samples,count=256,batch_size=4):
    _,indexed=summarize_samples(samples,count,batch_size);paired=[]
    scopes=[(str(int(g)),(g,)) for g in cfg.GAMMAS]+[
        ('low_0_5_10',cfg.GAMMAS[:3]),('all_six',cfg.GAMMAS)]
    for contrast in contrasts():
        for estimand,gammas in scopes:
            for metric in METRICS:
                values=[statistics.fmean(sum(c*indexed[(*key,g)][i][metric]
                    for key,c in contrast['coefficients'].items()) for g in gammas) for i in range(count)]
                batches=[statistics.fmean(values[i:i+batch_size]) for i in range(0,count,batch_size)]
                row=dict(contrast=contrast['name'],energy_policy=contrast['policy'],estimand=estimand,
                    gamma_points=','.join(str(int(g)) for g in gammas),metric=metric,seed=cfg.SEED,
                    num_samples=count,num_batches=len(batches),batch_size=batch_size,
                    coefficients=json.dumps([dict(arm=k[0],group_count=k[1],energy_policy=k[2],coefficient=v)
                        for k,v in contrast['coefficients'].items()],sort_keys=True),
                    sample_unit='one_complete_K8_channel',batch_unit='sample_index//4',
                    ci_scope='fixed-checkpoint single-seed validation sampling; not journal/test evidence',
                    interaction_from_raw_samples=True,source_commit=cfg.SOURCE_COMMIT)
                for prefix,stats in (('sample_',statistics_for(values)),('batch_',statistics_for(batches))):
                    row.update({prefix+k:v for k,v in stats.items()})
                row['batch_ci_estimable']=len(batches)>1
                paired.append(row)
    return paired


def _write_json(path,value):
    with path.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def _write_csv(path,rows,compressed=False):
    with (gzip.open if compressed else open)(path,'wt',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def artifact_hashes(directory):
    return {name:cfg.sha256_file(Path(directory)/name) for name in ARTIFACTS}


def _output(directory):
    path=cfg.inside_root(directory)
    if path.exists():raise FileExistsError('New isolated evaluation directory required')
    path.mkdir(parents=True,exist_ok=False);return path


def _tensor_sha(tensor):
    value=tensor.detach().contiguous().cpu()
    digest=hashlib.sha256((str(value.dtype)+str(tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes());return digest.hexdigest()


def _state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode());digest.update(_tensor_sha(value).encode())
    return digest.hexdigest()


def _snapshot(module):
    return dict(buffers={n:_tensor_sha(v) for n,v in module.named_buffers()},
        modes={n:m.training for n,m in module.named_modules()},versions={n:p._version for n,p in module.named_parameters()})


def run_evaluation(checkpoints,output_dir,*,device='cuda:0'):
    if set(checkpoints)!=set(cfg.ARMS) or device!='cuda:0':raise ValueError('Four explicit checkpoints and CUDA:0 required')
    output=_output(output_dir)
    manifest=dict(status='running',role='validation',test_used=False,optimizer_updates=0,seed=cfg.SEED,
        checkpoint_selection='predeclared final3024; no validation reselection',source_commit=cfg.SOURCE_COMMIT,
        samples=256,batch_size=4,validation_batch_addresses=list(range(2048,2112)),
        gamma_db=list(cfg.GAMMAS),pilot_energy_per_ue=16.,downlink_snr_db=25.,energy_ratios=POLICIES,
        shared_inputs=[],methods={},interpretation='ordered encoder single-seed controlled screen; unequal parameters')
    inputs={};chains={};ce_states={};failure=None
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        import train_twc_v1_k8_codec_warmup as old
        from train_twc_v1_k8_ordered_encoder_screen import (
            verify_completed_payload,source_code_record,ce_identity,build_chain,make_data)
        code=source_code_record();ce=ce_identity()
        manifest.update(experiment_commit=code['commit'],source_sha256=code['sha256'],ce_identity=ce,
            backend=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                matmul_precision=torch.get_float32_matmul_precision(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        for record in ce.values():inputs[record['path']]=cfg.sha256_file(record['path'])
        states={};state_hashes={};data_protocol=None;parameters=[]
        for arm in cfg.ARMS:
            path=Path(checkpoints[arm]).resolve();inputs[str(path)]=cfg.sha256_file(path)
            payload=torch.load(path,map_location='cpu',weights_only=False)
            verify_completed_payload(torch,payload,arm)
            if payload['ce_identity']!=ce:raise ValueError('CE/calibration identity changed')
            if data_protocol is None:data_protocol=payload['data_protocol']
            elif cfg.normalize_json(data_protocol)!=cfg.normalize_json(payload['data_protocol']):
                raise ValueError('Training data protocols differ')
            ordered=cfg.is_ordered(arm)
            if ordered not in chains:
                chains[ordered]=build_chain(torch,torch.device(device),arm,payload,initializing_source=False).eval()
                ce_states[ordered]=_state_sha(chains[ordered].ce_model.state_dict())
            chain=chains[ordered];expected=chain.codec_model.state_dict();state=payload['codec_model']
            if set(state)!=set(expected):raise ValueError('Complete trained state required; migration forbidden during evaluation')
            for name,value in state.items():
                if value.shape!=expected[name].shape or value.dtype!=expected[name].dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError('Trained state shape/dtype/finite mismatch: '+name)
            chain.codec_model.load_state_dict(state,strict=True);chain.eval()
            states[arm]=state;state_hashes[arm]=_state_sha(state)
            encoder=sum(p.numel() for p in chain.codec_model.encoder.parameters())
            full=sum(p.numel() for p in chain.codec_model.parameters())
            parameter=dict(arm=arm,encoder_parameters=encoder,codec_parameters=full,bs_parameters=full-encoder,
                ce_parameters=sum(p.numel() for p in chain.ce_model.parameters()),
                head_input_width=chain.codec_model.encoder.key_head.in_features,
                note='Actual stored numel, not equal-parameter/MAC/latency comparison')
            parameters.append(parameter)
            manifest['methods'][arm]=dict(path=str(path),sha256=inputs[str(path)],step=cfg.UPDATES,
                cumulative_updates=cfg.SOURCE_UPDATES+cfg.UPDATES,source=payload['source'],
                architecture=cfg.architecture(arm),strict_loading=True,migration_during_evaluation=False,
                source_commit=cfg.SOURCE_COMMIT,experiment_commit=code['commit'],parameters=parameter)
            del payload
        by_arm={r['arm']:r for r in parameters}
        for mean,ordered in (('mean_fdma_control','ordered_fdma'),('mean_g4_control','ordered_g4')):
            for key in ('codec_parameters','encoder_parameters'):
                if by_arm[ordered][key]-by_arm[mean][key]!=cfg.EXPECTED_PARAMETER_DELTA:
                    raise ValueError('Ordered parameter delta differs from the two head widths')
            if by_arm[ordered]['bs_parameters']!=by_arm[mean]['bs_parameters']:raise ValueError('BS capacity changed')
        data=make_data(old,torch.device(device))
        if cfg.normalize_json(data.protocol)!=cfg.normalize_json(data_protocol):raise ValueError('Actual validation data identity differs')
        manifest['data_protocol']=data.protocol;raw=[];torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for bi in range(cfg.VALIDATION_BATCHES):
                sample=data.sample('validation',cfg.VALIDATION_START_INDEX+bi);hdl,hul=sample['H_dl'],sample['H_ul']
                originals=(_tensor_sha(hdl),_tensor_sha(hul))
                shared=dict(batch_index=bi,address=sample['address'],H_dl_sha256=originals[0],H_ul_sha256=originals[1],noise={})
                noises={}
                for gi,gamma in enumerate(cfg.GAMMAS):
                    values=[]
                    for kind in ('pilot','data'):
                        gen=torch.Generator(device='cpu').manual_seed(cfg.evaluation_noise_seed(bi,gi,kind))
                        noise=torch.view_as_complex(torch.randn(cfg.BATCH_SIZE,32,256,2,generator=gen)).to(device)
                        values.append(noise);shared['noise'][f'{gamma}_{kind}']=_tensor_sha(noise)
                    noises[gamma]=tuple(values)
                ul_reference={}
                for arm in cfg.ARMS:
                    chain=chains[cfg.is_ordered(arm)];chain.codec_model.load_state_dict(states[arm],strict=True);chain.eval()
                    before=_snapshot(chain)
                    for _,g,policy,gamma in (c for c in conditions() if c[0]==arm):
                        if cfg.is_fdma(arm):layout,split=old.protocol.FDMALayout(),None
                        else:
                            ratio=POLICIES[policy];layout=old.protocol.default_hybrid_layout(4)
                            split=old.protocol.EnergySplit(16.,48.*ratio[0]/sum(ratio),48.*ratio[1]/sum(ratio))
                        pilot_noise,data_noise=noises[gamma]
                        out=chain(hdl,hul,feedback_gamma_db=gamma,pilot_energy_per_ue=16.,
                            pilot_standard_noise=pilot_noise,data_standard_noise=data_noise,layout=layout,energy_split=split)
                        ul_hash=_tensor_sha(out['H_ul_est'])
                        if ul_reference.setdefault(gamma,ul_hash)!=ul_hash:raise ValueError('CE estimate differs across methods')
                        metrics=chain.metrics(out,hdl,rzf_alpha=.01,downlink_snr_db=25.)
                        energy=out['energy']['actual_data_per_ue']
                        values=dict(dl_nmse=metrics['dl_nmse_per_user'].mean(-1),sum_rate_bps_hz=metrics['sum_rate_bps_hz'],
                            desired_signal_power=metrics['desired_signal_power_mean'],interference_power=metrics['interference_power_mean'],
                            data_energy_per_ue=energy.mean(-1),frame_energy_per_ue=energy.mean(-1)+16.,
                            energy_max_abs_error=(energy-48.).abs().amax(-1),
                            rzf_power_mean=metrics['rzf_per_subcarrier_power'].mean(-1),
                            rzf_power_max_abs_error=(metrics['rzf_per_subcarrier_power']-1.).abs().amax(-1))
                        arrays={k:v.detach().cpu().tolist() for k,v in values.items()}
                        raw.extend(dict(arm=arm,seed=cfg.SEED,group_count=g,energy_policy=policy,gamma_db=gamma,
                            pilot_energy_per_ue=16.,downlink_snr_db=25.,sample_index=bi*4+i,batch_index=bi,
                            channel_fingerprint=sample['sample_fingerprint'],checkpoint_step=cfg.UPDATES,
                            cumulative_updates=cfg.SOURCE_UPDATES+cfg.UPDATES,**{k:v[i] for k,v in arrays.items()}) for i in range(4))
                    if before!=_snapshot(chain):raise ValueError('Evaluation changed buffers/modes/parameter versions')
                    if bi in (0,cfg.VALIDATION_BATCHES-1) and _state_sha(chain.codec_model.state_dict())!=state_hashes[arm]:
                        raise ValueError('Evaluation changed trained model state')
                if originals!=(_tensor_sha(hdl),_tensor_sha(hul)):raise ValueError('Shared channel tensors changed')
                for gamma,pair in noises.items():
                    for kind,noise in zip(('pilot','data'),pair):
                        if _tensor_sha(noise)!=shared['noise'][f'{gamma}_{kind}']:raise ValueError('Shared noise changed')
                manifest['shared_inputs'].append(shared)
                print(f'Ordered encoder paired validation {bi+1}/{cfg.VALIDATION_BATCHES}',flush=True)
        summary,_=summarize_samples(raw,cfg.VALIDATION_SAMPLES,cfg.BATCH_SIZE)
        paired=paired_statistics(raw,cfg.VALIDATION_SAMPLES,cfg.BATCH_SIZE)
        _write_csv(output/'summary.csv',summary);_write_csv(output/'samples.csv.gz',raw,True)
        _write_csv(output/'paired_deltas.csv',paired);_write_csv(output/'parameters.csv',parameters)
        _write_json(output/'results.json',dict(summary=summary,paired=paired,parameters=parameters))
        manifest.update(status='completed',summary_rows=len(summary),sample_rows=len(raw),paired_rows=len(paired),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),results_verified=True)
    except BaseException as error:
        failure=error;manifest.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        errors=[]
        for path,digest in inputs.items():
            try:
                if cfg.sha256_file(path)!=digest:errors.append('Input changed: '+path)
            except OSError as error:errors.append('Unreadable input: '+path+': '+str(error))
        for path,digest in manifest.get('source_sha256',{}).items():
            try:
                if cfg.sha256_file(cfg.REPO_ROOT/path)!=digest:errors.append('Code changed: '+path)
            except OSError as error:errors.append('Unreadable code: '+path+': '+str(error))
        for kind,chain in chains.items():
            if _state_sha(chain.ce_model.state_dict())!=ce_states[kind]:errors.append('Frozen CE state changed')
        manifest.update(input_sha256=inputs,freeze_checks=dict(passed=not errors and len(chains)==2,errors=errors),
            checkpoint_files_unchanged=not any('input' in e.lower() for e in errors))
        if errors:
            failure=failure or RuntimeError('; '.join(errors));manifest.update(status='failed',error=str(failure))
        if failure is None:manifest['output_sha256']=artifact_hashes(output)
        _write_json(output/'manifest.json',manifest)
    if failure is not None:raise RuntimeError(f'Ordered encoder validation failed; evidence retained at {output}') from failure
    return dict(output_directory=str(output),manifest_path=str(output/'manifest.json'),summary=summary,paired=paired,
        parameters=parameters,artifacts=manifest['output_sha256'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--evaluate',action='store_true')
    parser.add_argument('--checkpoint',action='append',default=[],metavar='ARM=PATH');parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(arms=list(cfg.ARMS),conditions=len(conditions()),samples=256,batch_size=4,
            validation_start_index=2048,role='validation',test_used=False,source_commit=cfg.SOURCE_COMMIT,
            local_step=3024,cumulative_updates=16048,models_loaded=False,
            comparisons=[r['name'] for r in contrasts()]),indent=2));return 0
    if not args.evaluate:parser.print_help();return 0
    checkpoints={}
    for value in args.checkpoint:
        arm,sep,path=value.partition('=')
        if not sep or arm in checkpoints:parser.error('Distinct explicit ARM=PATH checkpoints required')
        checkpoints[arm]=path
    if args.output_dir is None:parser.error('--output-dir required')
    run_evaluation(checkpoints,args.output_dir);return 0


if __name__=='__main__':raise SystemExit(main())
