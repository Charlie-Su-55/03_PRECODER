#!/usr/bin/env python3
"""Shared-channel validation of four task-loss continuations and two frozen sources."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import traceback

from configs import twc_v1_k8_task_loss_screen_config as cfg
from evaluate_twc_v1_k8_ordered_encoder_screen import (
    statistics_for, _write_json, _write_csv, _tensor_sha, _state_sha, _snapshot)

METRICS=('dl_nmse','sum_rate_bps_hz','desired_signal_power','interference_power')
USER_RATE_METRICS=tuple(f'user_rate_ue{k}_bps_hz' for k in range(8))
DIAGNOSTICS=('data_energy_per_ue','frame_energy_per_ue','rzf_power_mean')
POLICIES={'hybrid_1_3':(1,3),'hybrid_1_1':(1,1)}
ARTIFACTS=('summary.csv','samples.csv.gz','paired_deltas.csv','parameters.csv','results.json')


def method_is_fdma(method):
    if method not in cfg.METHODS:raise ValueError('Unknown task-loss evaluation method')
    return method in ('fdma_nmse','fdma_joint','source_fdma')


def method_steps(method):
    method_is_fdma(method)
    return ((cfg.SOURCE_LOCAL_UPDATES,cfg.SOURCE_UPDATES) if method.startswith('source_')
            else (cfg.UPDATES,cfg.SOURCE_UPDATES+cfg.UPDATES))


def conditions():
    return tuple((method,1 if method_is_fdma(method) else 4,policy,gamma)
        for method in cfg.METHODS for policy in (('fdma',) if method_is_fdma(method) else tuple(POLICIES))
        for gamma in cfg.GAMMAS)


def summarize_samples(samples,count=256,batch_size=4):
    """Require all54 conditions and complete MU sample identities before statistics."""
    if type(count) is not int or count<batch_size or count%batch_size or batch_size!=4:
        raise ValueError('Complete batches of four multi-user channel samples required')
    indexed={condition:{} for condition in conditions()};fingerprints={}
    for row in samples:
        key=tuple(row.get(k) for k in ('arm','group_count','energy_policy','gamma_db'))
        i=row.get('sample_index')
        if key not in indexed or type(i) is not int or not 0<=i<count or i in indexed[key]:
            raise ValueError('Unexpected or duplicate complete task-loss sample key')
        step,cumulative=method_steps(key[0])
        if (row.get('seed')!=cfg.SEED or row.get('checkpoint_step')!=step
                or row.get('cumulative_updates')!=cumulative or row.get('batch_index')!=i//batch_size
                or row.get('pilot_energy_per_ue')!=16. or row.get('downlink_snr_db')!=25.):
            raise ValueError('Sample protocol/source/continuation identity changed')
        fingerprint=row.get('channel_fingerprint')
        if (not isinstance(fingerprint,str) or len(fingerprint)!=64
                or any(c not in '0123456789abcdef' for c in fingerprint)):
            raise ValueError('Actual common channel fingerprint required')
        if fingerprints.setdefault(i//batch_size,fingerprint)!=fingerprint:
            raise ValueError('Methods/conditions do not share actual channels')
        for metric in METRICS+USER_RATE_METRICS+DIAGNOSTICS:
            value=row.get(metric)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Nonfinite/negative metric: '+metric)
        if not math.isclose(sum(row[key] for key in USER_RATE_METRICS),row['sum_rate_bps_hz'],rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError('Per-user rate columns do not sum to the MU sum rate')
        for metric,tolerance in (('energy_max_abs_error',1e-4),('rzf_power_max_abs_error',1e-5)):
            value=row.get(metric)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=tolerance:
                raise ValueError('Energy/RZF constraint failed: '+metric)
        if (abs(row['data_energy_per_ue']-48.)>1e-4 or abs(row['frame_energy_per_ue']-64.)>1e-4
                or abs(row['rzf_power_mean']-1.)>1e-5):raise ValueError('Energy or RZF power units changed')
        indexed[key][i]=row
    summary=[]
    for key,records in indexed.items():
        if set(records)!=set(range(count)):raise ValueError('Incomplete method/energy/gamma/sample coverage')
        row=dict(zip(('arm','group_count','energy_policy','gamma_db'),key))
        step,cumulative=method_steps(key[0])
        row.update(seed=cfg.SEED,num_samples=count,num_batches=count//batch_size,batch_size=batch_size,
            checkpoint_step=step,cumulative_updates=cumulative,source_commit=cfg.SOURCE_COMMIT,
            role='validation',test_used=False,method_type='frozen_source' if key[0].startswith('source_') else 'continued_'+key[0].split('_')[-1])
        for metric in METRICS+USER_RATE_METRICS+DIAGNOSTICS:
            row.update({metric+'_'+name:value for name,value in
                statistics_for([records[i][metric] for i in range(count)]).items()})
        row['energy_max_abs_error']=max(record['energy_max_abs_error'] for record in records.values())
        row['rzf_power_max_abs_error']=max(record['rzf_power_max_abs_error'] for record in records.values())
        summary.append(row)
    return summary,indexed


def contrasts():
    """Raw-sample coefficients: task-loss effects, matched gaps, and frozen controls."""
    result=[]
    for name,left,right in (
            ('fdma_joint_minus_fdma_nmse','fdma_joint','fdma_nmse'),
            ('fdma_nmse_minus_source_fdma','fdma_nmse','source_fdma'),
            ('fdma_joint_minus_source_fdma','fdma_joint','source_fdma')):
        result.append(dict(name=name,policy='fdma',coefficients={(left,1,'fdma'):1,(right,1,'fdma'):-1}))
    for policy in POLICIES:
        gj=('g4_joint',4,policy);gn=('g4_nmse',4,policy);gs=('source_g4',4,policy)
        fj=('fdma_joint',1,'fdma');fn=('fdma_nmse',1,'fdma')
        for name,coefficients in (
                ('g4_joint_minus_g4_nmse',{gj:1,gn:-1}),
                ('g4_joint_minus_fdma_joint',{gj:1,fj:-1}),
                ('gap_change',{gj:1,fj:-1,gn:-1,fn:1}),
                ('g4_nmse_minus_source_g4',{gn:1,gs:-1}),
                ('g4_joint_minus_source_g4',{gj:1,gs:-1})):
            result.append(dict(name=name+'_'+policy,policy=policy,coefficients=coefficients))
    return result


def paired_statistics(samples,count=256,batch_size=4):
    _,indexed=summarize_samples(samples,count,batch_size);paired=[]
    scopes=[(str(int(g)),(g,)) for g in cfg.GAMMAS]+[
        ('low_0_5_10',cfg.GAMMAS[:3]),('all_six',cfg.GAMMAS)]
    for contrast in contrasts():
        for estimand,gammas in scopes:
            for metric in METRICS:
                values=[statistics.fmean(sum(coefficient*indexed[(*key,gamma)][i][metric]
                    for key,coefficient in contrast['coefficients'].items()) for gamma in gammas) for i in range(count)]
                batches=[statistics.fmean(values[i:i+batch_size]) for i in range(0,count,batch_size)]
                row=dict(contrast=contrast['name'],energy_policy=contrast['policy'],estimand=estimand,
                    gamma_points=','.join(str(int(g)) for g in gammas),metric=metric,seed=cfg.SEED,
                    num_samples=count,num_batches=len(batches),batch_size=batch_size,
                    coefficients=json.dumps([dict(arm=key[0],group_count=key[1],energy_policy=key[2],coefficient=value)
                        for key,value in contrast['coefficients'].items()],sort_keys=True),
                    sample_unit='one_complete_K8_channel',batch_unit='sample_index//4',
                    ci_scope='fixed checkpoints, single training seed, validation sampling; not formal held-out test evidence',
                    interaction_from_raw_samples=True,source_commit=cfg.SOURCE_COMMIT)
                for prefix,stats in (('sample_',statistics_for(values)),('batch_',statistics_for(batches))):
                    row.update({prefix+name:value for name,value in stats.items()})
                row['batch_ci_estimable']=len(batches)>1;paired.append(row)
    return paired


def artifact_hashes(directory):
    return {name:cfg.sha256_file(Path(directory)/name) for name in ARTIFACTS}


def validate_checkpoint_paths(checkpoints):
    if set(checkpoints)!=set(cfg.ARMS):raise ValueError('Four explicit task-loss terminal checkpoints required')
    paths={}
    for arm,path in checkpoints.items():
        path=cfg.inside_root(path)
        if path!=(cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt').resolve():
            raise ValueError('Only this arm prescribed task-loss terminal checkpoint is allowed: '+arm)
        paths[arm]=path
    return paths


def _output(directory):
    path=cfg.inside_root(directory)
    if path.exists():raise FileExistsError('New isolated task-loss validation directory required')
    path.mkdir(parents=True,exist_ok=False);return path


def _source_arm(method):
    return 'fdma_nmse' if method_is_fdma(method) else 'g4_nmse'


def _include_evidence_files(inputs,evidence):
    """Retain hashes supplied by the authoritative loader; never infer substitute paths."""
    if not isinstance(evidence,dict):return
    for key in ('frozen_files','input_sha256'):
        for path,digest in evidence.get(key,{}).items():
            if path in inputs and inputs[path]!=digest:raise ValueError('Conflicting frozen input identity: '+path)
            inputs[path]=digest


def run_evaluation(checkpoints,output_dir,*,device='cuda:0'):
    if device!='cuda:0':raise ValueError('CUDA:0 is required, without CPU fallback')
    checkpoints=validate_checkpoint_paths(checkpoints)
    calibration=json.loads(cfg.CALIBRATION_PATH.read_text(encoding='utf-8'))
    calibration_identity=cfg.calibration_identity(calibration)
    output=_output(output_dir)
    manifest=dict(status='running',role='validation',test_used=False,optimizer_updates=0,seed=cfg.SEED,
        checkpoint_selection='predeclared final6048, frozen source3024; no validation reselection',
        source_commit=cfg.SOURCE_COMMIT,internal_source_commit=cfg.INTERNAL_SOURCE_COMMIT,
        samples=256,batch_size=4,validation_batch_addresses=list(range(3072,3136)),
        gamma_db=list(cfg.GAMMAS),pilot_energy_per_ue=16.,downlink_snr_db=25.,energy_ratios=POLICIES,
        screen_version=cfg.NAMESPACE,calibration=calibration_identity,shared_inputs=[],methods={},
        interpretation='same ordered encoder, isolated NMSE/task-loss continuations, one seed; frozen sources retain earlier budget')
    inputs={str(cfg.CALIBRATION_PATH):cfg.sha256_file(cfg.CALIBRATION_PATH)}
    chains={};ce_states={};failure=None
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        import train_twc_v1_k8_codec_warmup as warmup
        from train_twc_v1_k8_task_loss_screen import (
            validate_calibration_record,load_final_payload,load_sources,build_chain,source_code_record,ce_identity,make_data)
        validate_calibration_record(calibration)
        code=source_code_record();ce=ce_identity();data=make_data(warmup,torch.device(device))
        if (calibration.get('code')!=code
                or cfg.normalize_json(calibration.get('data_protocol'))!=cfg.normalize_json(data.protocol)
                or calibration.get('ce_identity')!=ce):
            raise ValueError('Calibration experiment code/data/CE identity differs from this evaluation')
        _include_evidence_files(inputs,calibration)
        manifest.update(experiment_commit=code['commit'],source_sha256=code['sha256'],ce_identity=ce,data_protocol=data.protocol,
            backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                matmul_precision=torch.get_float32_matmul_precision(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        for record in ce.values():inputs[record['path']]=cfg.sha256_file(record['path'])
        states={};state_hashes={};parameters=[]
        sources=load_sources(torch,warmup,data.protocol)
        if set(sources)!={'source_fdma','source_g4'}:raise ValueError('Both exact frozen source identities are required')
        for method in cfg.METHODS:
            if method.startswith('source_'):
                payload,evidence=sources.pop(method)
                identity=cfg.source_identity(_source_arm(method));path=Path(identity['path'])
                expected_sha=identity['sha256']
                if cfg.sha256_file(path)!=expected_sha:raise ValueError('Frozen source checkpoint hash changed')
            else:
                path=checkpoints[method]
                payload,evidence=load_final_payload(torch,method,path,calibration,data.protocol,ce)
                expected_sha=cfg.sha256_file(path)
            inputs[str(path)]=expected_sha;_include_evidence_files(inputs,evidence)
            family=method_is_fdma(method)
            if family not in chains:
                chains[family]=build_chain(torch,torch.device(device),method,payload).eval()
                ce_states[family]=_state_sha(chains[family].ce_model.state_dict())
            chain=chains[family];expected=chain.codec_model.state_dict();state=payload['codec_model']
            if not isinstance(state,dict) or set(state)!=set(expected):
                raise ValueError('Exact trained ordered state required; migration and gate resetting forbidden')
            for name,value in state.items():
                if (not torch.is_tensor(value) or value.shape!=expected[name].shape
                        or value.dtype!=expected[name].dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError('Final state tensor shape/dtype/finite mismatch: '+name)
            chain.codec_model.load_state_dict(state,strict=True);chain.eval()
            states[method]=state;state_hashes[method]=_state_sha(state)
            ue=chain.codec_model.encoder
            if ue.key_head.in_features!=128 or ue.shared_head.in_features!=128:
                raise ValueError('Required unchanged ordered-v2 mean heads are missing')
            residuals=[getattr(ue,key,None) for key in ('key_window_residual','shared_window_residual')]
            residual_hashes={}
            for name,module,out in zip(('key_window_residual','shared_window_residual'),residuals,(8,320)):
                if module is None or module.in_features!=3968 or module.out_features!=out or module.bias is not None:
                    raise ValueError('Required trained ordered-v2 residual shape is missing')
                residual_hashes[name+'.weight']=_tensor_sha(module.weight)
            encoder=sum(parameter.numel() for parameter in ue.parameters())
            full=sum(parameter.numel() for parameter in chain.codec_model.parameters())
            parameter=dict(arm=method,encoder_parameters=encoder,codec_parameters=full,bs_parameters=full-encoder,
                ce_parameters=sum(parameter.numel() for parameter in chain.ce_model.parameters()),
                ordered_residual_parameters=sum(parameter.numel() for module in residuals for parameter in module.parameters()),
                architecture_id=cfg.ORDERED_ARCHITECTURE,note='Actual stored numel; no MAC or latency claim')
            parameters.append(parameter);step,cumulative=method_steps(method)
            manifest['methods'][method]=dict(path=str(path),sha256=expected_sha,step=step,cumulative_updates=cumulative,
                source=payload['source'],strict_loading=True,migration_during_evaluation=False,
                trained_residual_weight_sha256=residual_hashes,residuals_reinitialized=False,
                parameters=parameter,loading_evidence=evidence,
                calibration=payload.get('calibration') if method in cfg.ARMS else None)
            del payload
        if sources:raise ValueError('Unused frozen source payloads remain')
        if len({tuple(row[key] for key in ('encoder_parameters','codec_parameters','bs_parameters')) for row in parameters})!=1:
            raise ValueError('All six task-loss methods must retain the same stored architecture')
        raw=[];torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for bi in range(cfg.VALIDATION_BATCHES):
                sample=data.sample('validation',cfg.VALIDATION_START_INDEX+bi);hdl,hul=sample['H_dl'],sample['H_ul']
                originals=(_tensor_sha(hdl),_tensor_sha(hul))
                shared=dict(batch_index=bi,address=sample['address'],H_dl_sha256=originals[0],H_ul_sha256=originals[1],noise={})
                noises={}
                for gi,gamma in enumerate(cfg.GAMMAS):
                    values=[]
                    for kind in ('pilot','data'):
                        generator=torch.Generator(device='cpu').manual_seed(cfg.evaluation_noise_seed(bi,gi,kind))
                        noise=torch.view_as_complex(torch.randn(cfg.BATCH_SIZE,32,256,2,generator=generator)).to(device)
                        values.append(noise);shared['noise'][f'{gamma}_{kind}']=_tensor_sha(noise)
                    noises[gamma]=tuple(values)
                ul_reference={}
                for method in cfg.METHODS:
                    chain=chains[method_is_fdma(method)];chain.codec_model.load_state_dict(states[method],strict=True);chain.eval()
                    before=_snapshot(chain)
                    for _,g,policy,gamma in (condition for condition in conditions() if condition[0]==method):
                        if method_is_fdma(method):layout,split=warmup.protocol.FDMALayout(),None
                        else:
                            ratio=POLICIES[policy];layout=warmup.protocol.default_hybrid_layout(4)
                            split=warmup.protocol.EnergySplit(16.,48.*ratio[0]/sum(ratio),48.*ratio[1]/sum(ratio))
                        pilot_noise,data_noise=noises[gamma]
                        out=chain(hdl,hul,feedback_gamma_db=gamma,pilot_energy_per_ue=16.,
                            pilot_standard_noise=pilot_noise,data_standard_noise=data_noise,layout=layout,energy_split=split)
                        ul_hash=_tensor_sha(out['H_ul_est'])
                        if ul_reference.setdefault(gamma,ul_hash)!=ul_hash:raise ValueError('Frozen CE estimate differs across methods')
                        metrics=chain.metrics(out,hdl,rzf_alpha=.01,downlink_snr_db=25.)
                        if tuple(metrics['user_rate_bps_hz'].shape)!=(cfg.BATCH_SIZE,8):
                            raise ValueError('Eight per-user rates for each complete MU sample are required')
                        energy=out['energy']['actual_data_per_ue']
                        values=dict(dl_nmse=metrics['dl_nmse_per_user'].mean(-1),sum_rate_bps_hz=metrics['sum_rate_bps_hz'],
                            desired_signal_power=metrics['desired_signal_power_mean'],interference_power=metrics['interference_power_mean'],
                            data_energy_per_ue=energy.mean(-1),frame_energy_per_ue=energy.mean(-1)+16.,
                            energy_max_abs_error=(energy-48.).abs().amax(-1),
                            rzf_power_mean=metrics['rzf_per_subcarrier_power'].mean(-1),
                            rzf_power_max_abs_error=(metrics['rzf_per_subcarrier_power']-1.).abs().amax(-1),
                            **{name:metrics['user_rate_bps_hz'][:,k] for k,name in enumerate(USER_RATE_METRICS)})
                        arrays={key:value.detach().cpu().tolist() for key,value in values.items()}
                        step,cumulative=method_steps(method)
                        raw.extend(dict(arm=method,seed=cfg.SEED,group_count=g,energy_policy=policy,gamma_db=gamma,
                            pilot_energy_per_ue=16.,downlink_snr_db=25.,sample_index=bi*4+i,batch_index=bi,
                            channel_fingerprint=sample['sample_fingerprint'],checkpoint_step=step,cumulative_updates=cumulative,
                            **{key:value[i] for key,value in arrays.items()}) for i in range(4))
                    if before!=_snapshot(chain):raise ValueError('Evaluation changed buffers/modes/parameter versions')
                    if bi in (0,cfg.VALIDATION_BATCHES-1) and _state_sha(chain.codec_model.state_dict())!=state_hashes[method]:
                        raise ValueError('Evaluation changed trained model state')
                if originals!=(_tensor_sha(hdl),_tensor_sha(hul)):raise ValueError('Shared channel tensors changed')
                for gamma,pair in noises.items():
                    for kind,noise in zip(('pilot','data'),pair):
                        if _tensor_sha(noise)!=shared['noise'][f'{gamma}_{kind}']:raise ValueError('Shared standard noise changed')
                manifest['shared_inputs'].append(shared)
                print(f'Task-loss paired validation {bi+1}/{cfg.VALIDATION_BATCHES}',flush=True)
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
        for family,chain in chains.items():
            if _state_sha(chain.ce_model.state_dict())!=ce_states[family]:errors.append('Frozen CE state changed')
        manifest.update(input_sha256=inputs,freeze_checks=dict(passed=not errors and len(chains)==2,errors=errors),
            checkpoint_files_unchanged=not any('input' in error.lower() for error in errors))
        if errors:
            failure=failure or RuntimeError('; '.join(errors));manifest.update(status='failed',error=str(failure))
        if failure is None:manifest['output_sha256']=artifact_hashes(output)
        _write_json(output/'manifest.json',manifest)
    if failure is not None:raise RuntimeError(f'Task-loss validation failed; evidence retained at {output}') from failure
    return dict(output_directory=str(output),manifest_path=str(output/'manifest.json'),summary=summary,paired=paired,
        parameters=parameters,artifacts=manifest['output_sha256'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--evaluate',action='store_true')
    parser.add_argument('--checkpoint',action='append',default=[],metavar='ARM=PATH');parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(methods=list(cfg.METHODS),conditions=len(conditions()),samples=256,batch_size=4,
            validation_start_index=cfg.VALIDATION_START_INDEX,role='validation',test_used=False,
            source_implementation_commit=cfg.SOURCE_COMMIT,screen_version=cfg.NAMESPACE,
            trained_local_step=cfg.UPDATES,trained_cumulative_updates=cfg.SOURCE_UPDATES+cfg.UPDATES,
            source_local_step=cfg.SOURCE_LOCAL_UPDATES,source_cumulative_updates=cfg.SOURCE_UPDATES,
            calibration=dict(path=str(cfg.CALIBRATION_PATH),loaded=False),models_loaded=False,
            comparisons=[record['name'] for record in contrasts()],per_user_rate_columns=list(USER_RATE_METRICS)),indent=2));return 0
    if not args.evaluate:parser.print_help();return 0
    checkpoints={}
    for value in args.checkpoint:
        arm,separator,path=value.partition('=')
        if not separator or arm in checkpoints:parser.error('Distinct explicit ARM=PATH checkpoints required')
        checkpoints[arm]=path
    if args.output_dir is None:parser.error('--output-dir required')
    run_evaluation(checkpoints,args.output_dir);return 0


if __name__=='__main__':raise SystemExit(main())
