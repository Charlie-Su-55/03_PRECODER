#!/usr/bin/env python3
"""Archive-bound inference-only completion of the accepted physical-split batch.

Old statistical/protocol helpers are reused unchanged. Weight validation and
model construction belong exclusively to the new finish-validation boundary.
No old trainer loader, current-HEAD validator, or mutable global is patched.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import traceback

import evaluate_twc_v1_k8_physical_split as old
import physical_split_finish_validation as validation
from physical_split_finish_io import write_union_csv

cfg=old.cfg
ROOT=validation.ROOT
OUTPUT_ROOT=validation.OUTPUT_ROOT
NAMESPACE='twc_v1_k8_physical_split_finish_v1'
ACCEPTANCE_PATH=OUTPUT_ROOT/'training_acceptance.json'
SELECTION_PATH=OUTPUT_ROOT/'alpha_selection.json'
ATTEMPT_PATH=OUTPUT_ROOT/'selection_attempt.json'
SELECTION_FORMAT='twc_v1_k8_physical_split_finish_alpha_selection_v1'
ARTIFACTS=old.ARTIFACTS
METRICS,METRIC_UNITS,POLICIES=old.METRICS,old.METRIC_UNITS,old.POLICIES
base_conditions,conditions,contrasts=old.base_conditions,old.conditions,old.contrasts
select_global_alphas,summarize_samples,paired_statistics=old.select_global_alphas,old.summarize_samples,old.paired_statistics
offline_diagnostics,expected_diagnostic_metrics,diagnostic_summary=old.offline_diagnostics,old.expected_diagnostic_metrics,old.diagnostic_summary
score_cached_hhat,pilot_only_observation=old.score_cached_hhat,old.pilot_only_observation
method_steps,method_updates,rename_score=old.method_steps,old.method_updates,old.rename_score
_tensor_sha,_state_sha,_snapshot=old._tensor_sha,old._state_sha,old._snapshot
_frozen,canonical,sha256,read_json=old._frozen,old.canonical,old.sha256,old.read_json


def now():return datetime.now(timezone.utc).isoformat()
def training_code():return validation.archived_training_code()
def postprocessing_code_record():return validation.postprocessing_code_record()
def inside_root(path):
    target,allowed=Path(path).resolve(),OUTPUT_ROOT.resolve()
    if target==allowed or not target.is_relative_to(allowed):raise ValueError('New finish-root child path required')
    return target
def save_json(path,value):
    path=inside_root(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    temporary.replace(path)
def write_once_json(path,value):
    path=inside_root(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
def acceptance_identity(record):
    digest=record.get('record_id') if isinstance(record,dict) else None
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Immutable finish training-acceptance identity required')
    if cfg.json_sha256({key:value for key,value in record.items() if key!='record_id'})!=digest:
        raise ValueError('Finish training-acceptance record changed')
    return dict(format=record['format'],record_id=digest,training_commit=validation.TRAINING_COMMIT)
def selection_identity(record):
    selected=record.get('selected_alphas') if isinstance(record,dict) else None
    if (not isinstance(record,dict) or record.get('format')!=SELECTION_FORMAT or record.get('status')!='passed'
            or canonical(record.get('grid'))!=list(cfg.ALPHAS) or not isinstance(selected,dict) or set(selected)!=set(cfg.METHODS)):
        raise ValueError('Independent finish-root eight-checkpoint alpha lock required')
    for method,alpha in selected.items():
        if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or alpha not in cfg.ALPHAS:
            raise ValueError('Exactly one candidate global alpha required: '+method)
    digest=cfg.json_sha256({key:value for key,value in record.items() if key!='record_id'})
    if record.get('record_id')!=digest:raise ValueError('Immutable finish alpha selection changed')
    return dict(format=SELECTION_FORMAT,record_id=digest,selected_alphas={method:float(selected[method]) for method in cfg.METHODS})
def validate_selection_record(record):
    identity=selection_identity(record)
    for key,value in select_global_alphas(record.get('calibration_summary',[])).items():
        if canonical(record.get(key))!=canonical(value):raise ValueError('Locked alpha calibration decision changed: '+key)
    if (record.get('training_code')!=training_code() or record.get('postprocessing_code')!=postprocessing_code_record()
            or record.get('validation_addresses')!=list(range(15360,15376)) or len(record.get('shared_inputs',[]))!=16
            or set(record.get('methods',{}))!=set(cfg.METHODS) or not record.get('training_acceptance',{}).get('record_id')
            or not record.get('frozen_files') or record.get('freeze_checks',{}).get('passed') is not True
            or record.get('test_used') is not False):raise ValueError('Incomplete finish alpha archive/input evidence')
    return identity
def load_selection():
    record=read_json(SELECTION_PATH);validate_selection_record(record)
    acceptance=validation.load_acceptance()
    if record['training_acceptance']!=acceptance_identity(acceptance):raise ValueError('Locked finish training inputs changed')
    for path,digest in record['frozen_files'].items():
        if sha256(path)!=digest:raise ValueError('Locked finish input changed: '+path)
    directory=inside_root(record['calibration_directory'])
    if sha256(directory/'manifest.json')!=record['calibration_manifest_sha256']:raise ValueError('Finish calibration manifest changed')
    verify_stage_files(directory,'calibration');return record

def run_stage(stage,output_dir,selection=None):
    if stage not in ARTIFACTS:raise ValueError('Unknown decoder diagnostic split')
    output=inside_root(output_dir)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',stage=stage,namespace=NAMESPACE,protocol_namespace=cfg.NAMESPACE,training_code=training_code(),postprocessing_code=postprocessing_code_record(),test_used=False,optimizer_updates=0,metric_units=METRIC_UNITS,
        shared_inputs=[],network_forward_calls=0,rzf_solves=0,execution={})
    inputs={};failure=None;chains={};ce_before={}
    try:
        acceptance=validation.load_acceptance();manifest['training_acceptance']=acceptance_identity(acceptance)
        inputs[str(ACCEPTANCE_PATH)]=sha256(ACCEPTANCE_PATH);_frozen(inputs,acceptance)
        if stage=='evaluation':
            saved=load_selection();selection=saved if selection is None else selection
            if selection_identity(selection)!=selection_identity(saved):raise ValueError('Final alpha lock differs')
            inputs[str(SELECTION_PATH)]=sha256(SELECTION_PATH)
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        import train_twc_v1_k8_codec_warmup as warmup
        device=torch.device('cuda:0');data=validation.make_data(warmup,device);data.prepare();ce=validation.ce_identity()
        if acceptance['training_code']!=training_code() or acceptance['postprocessing_code']!=postprocessing_code_record() or canonical(acceptance['data_protocol'])!=canonical(data.protocol) or acceptance['ce_identity']!=ce:
            raise ValueError('Source acceptance code/data/CE identity mismatch')
        loaded=validation.load_methods(torch,warmup,data.protocol,ce,acceptance);chains=loaded['chains'];_frozen(inputs,loaded)
        for kind,chain in chains.items():ce_before[kind]=_state_sha(chain.ce_model.state_dict())
        for item in ce.values():inputs[item['path']]=item['sha256']
        manifest.update(data_protocol=data.protocol,ce_identity=ce,methods=loaded['methods'],parameters=loaded['parameters'],
            backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                matmul_precision=torch.get_float32_matmul_precision(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        if stage=='evaluation':
            if selection['training_acceptance']!=acceptance_identity(acceptance) or canonical(selection['data_protocol'])!=canonical(data.protocol):raise ValueError('Locked acceptance/data mismatch')
            for path,digest in selection['frozen_files'].items():
                if inputs.get(path)!=digest:raise ValueError('Locked checkpoint/input changed: '+path)
        batches=16 if stage=='calibration' else 64;start=15360 if stage=='calibration' else 16384
        manifest['validation_addresses']=list(range(start,start+batches))
        if stage=='calibration':write_once_json(OUTPUT_ROOT/'selection_attempt.json',dict(started=now(),postprocessing_code=postprocessing_code_record(),training_code=training_code(),grid=list(cfg.ALPHAS)))
        raw=[];diagnostics=[];totals={};torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for bi in range(batches):
                sample=data.sample('validation',start+bi);hdl,hul=sample['H_dl'],sample['H_ul'];originals=(_tensor_sha(hdl),_tensor_sha(hul));noises={};refs={}
                shared=dict(batch_index=bi,address=start+bi,H_dl_sha256=originals[0],H_ul_sha256=originals[1],sample_fingerprint=sample['sample_fingerprint'],noise={},ce_inputs={})
                for gi,gamma in enumerate(cfg.GAMMAS):
                    pair=[]
                    for kind in ('pilot','data'):
                        gen=torch.Generator(device='cpu').manual_seed(cfg.noise_seed('alpha_calibration' if stage=='calibration' else 'evaluation',bi,gi,kind))
                        noise=torch.view_as_complex(torch.randn(4,32,256,2,generator=gen)).to(device);pair.append(noise);shared['noise'][f'{gamma}_{kind}']=_tensor_sha(noise)
                    noises[gamma]=tuple(pair)
                pilot_indices=sorted({q for row in warmup.protocol.PILOT_POSITIONS_BY_USER for q in row})
                for method in cfg.METHODS:
                    chain=chains[cfg.variant(method)];chain.codec_model.load_state_dict(loaded['states'][method],strict=True);chain.eval();before=_snapshot(chain)
                    for _,g,policy,gamma in (c for c in base_conditions() if c[0]==method):
                        if cfg.is_fdma(method):layout,split=warmup.protocol.FDMALayout(),None
                        else:
                            ratio=POLICIES[policy];layout=warmup.protocol.default_hybrid_layout(4);split=warmup.protocol.EnergySplit(16.,48.*ratio[0]/sum(ratio),48.*ratio[1]/sum(ratio))
                        out=chain(hdl,hul,feedback_gamma_db=gamma,pilot_energy_per_ue=16.,pilot_standard_noise=noises[gamma][0],data_standard_noise=noises[gamma][1],layout=layout,energy_split=split)
                        manifest['network_forward_calls']+=1
                        ce_inputs=(pilot_only_observation(torch,out,pilot_indices),out['H_ul_ls'],out['H_ul_est'])
                        if gamma not in refs:
                            refs[gamma]=tuple(value.detach().clone() for value in ce_inputs);shared['ce_inputs'][str(gamma)]={name:_tensor_sha(value) for name,value in zip(('pilot_y','H_ls','H_ul_est'),ce_inputs)}
                        elif any(not torch.equal(left,right) for left,right in zip(refs[gamma],ce_inputs)):raise ValueError('Actual CE inputs/estimate differ')
                        scores={alpha:rename_score(value) for alpha,value in score_cached_hhat(torch,warmup._system.downlink,out['H_dl_hat'],hdl,cfg.ALPHAS if stage=='calibration' else (cfg.FIXED_ALPHA,selection['selected_alphas'][method])).items()}
                        manifest['rzf_solves']+=len(scores);energy=out['energy']['actual_data_per_ue'];error=(energy-48.).abs().amax(-1)
                        if not bool(torch.isfinite(energy).all()) or error.max().item()>1e-4:raise ValueError('UE energy mismatch')
                        execution={key:value for key,value in (out.get('receiver_execution') or {}).items() if key!='stages'}
                        execution=canonical(execution);ekey=method+'_'+policy+'_'+str(gamma)
                        if manifest['execution'].setdefault(ekey,execution)!=execution:raise ValueError('Receiver execution structure changed across batches')
                        if stage=='calibration':
                            for alpha,score in scores.items():
                                sums=totals.setdefault((method,g,policy,gamma,alpha),dict(count=0,rate=0.));sums['count']+=4;sums['rate']+=math.fsum(score['sum_rate_bps_hz'].cpu().tolist())
                        else:
                            offline=offline_diagnostics(out)
                            if set(offline)!=set(expected_diagnostic_metrics(method)):raise ValueError('Incomplete branch/stage offline diagnostic metrics')
                            for metric,value in offline.items():
                                if tuple(value.shape)!=(4,) or not bool(torch.isfinite(value).all()) or bool((value<0).any()):raise ValueError('Invalid offline receiver metric: '+metric)
                                diagnostics.extend(dict(arm=method,group_count=g,energy_policy=policy,gamma_db=gamma,metric=metric,sample_index=bi*4+i,batch_index=bi,channel_fingerprint=sample['sample_fingerprint'],value=float(v)) for i,v in enumerate(value.detach().cpu().tolist()))
                            step,cumulative=method_steps(method)
                            for mode,alpha in (('fixed',cfg.FIXED_ALPHA),('tuned',selection['selected_alphas'][method])):
                                arrays={name:value.cpu().tolist() for name,value in dict(scores[alpha],data_energy_per_ue=energy.mean(-1),frame_energy_per_ue=energy.mean(-1)+16.,energy_max_abs_error=error).items()}
                                counts=method_updates(method)
                                raw.extend(dict(arm=method,rzf_mode=mode,group_count=g,energy_policy=policy,gamma_db=gamma,alpha=alpha,seed=cfg.SEED,pilot_energy_per_ue=16.,downlink_snr_db=25.,sample_index=bi*4+i,batch_index=bi,channel_fingerprint=sample['sample_fingerprint'],checkpoint_step=step,data_global_step=cumulative,source_codec_updates=cfg.SOURCE_CODEC_UPDATES,source_bs_updates=cfg.SOURCE_BS_UPDATES,source_data_address=cfg.SOURCE_DATA_ADDRESS,update_counts=json.dumps(counts,sort_keys=True),ue_frozen=True,**counts,**{name:value[i] for name,value in arrays.items()}) for i in range(4))
                        del out,scores,ce_inputs
                    if before!=_snapshot(chain) or _state_sha(chain.codec_model.state_dict())!=loaded['methods'][method]['state_sha256']:raise ValueError('Frozen receiver state/buffers/modes changed')
                if originals!=(_tensor_sha(hdl),_tensor_sha(hul)):raise ValueError('Shared channel tensors changed')
                for gamma,pair in noises.items():
                    for kind,value in zip(('pilot','data'),pair):
                        if _tensor_sha(value)!=shared['noise'][f'{gamma}_{kind}']:raise ValueError('Shared standard noise changed')
                manifest['shared_inputs'].append(shared);del refs,noises
                print(f'Physical-split {stage} batch {bi+1}/{batches}',flush=True)
        if stage=='calibration':
            summary=[dict(zip(('arm','group_count','energy_policy','gamma_db','alpha'),key),num_samples=value['count'],sum_rate_bps_hz_mean=value['rate']/value['count']) for key,value in totals.items()]
            chosen=select_global_alphas(summary);write_union_csv(output/'alpha_calibration.csv',summary);write_union_csv(output/'alpha_global_scores.csv',chosen['scores']);manifest.update(calibration_summary=summary,**chosen)
        else:
            summary,_=summarize_samples(raw);paired=paired_statistics(raw);secondary=diagnostic_summary(diagnostics)
            write_union_csv(output/'summary.csv',summary);write_union_csv(output/'samples.csv.gz',raw,compressed=True);write_union_csv(output/'paired_deltas.csv',paired);write_union_csv(output/'parameters.csv',loaded['parameters'])
            write_union_csv(output/'receiver_diagnostics.csv.gz',diagnostics,compressed=True);write_union_csv(output/'receiver_diagnostic_summary.csv',secondary)
            save_json(output/'results.json',dict(summary=summary,paired=paired,parameters=loaded['parameters'],methods=loaded['methods'],receiver_diagnostics=secondary,selection=selection_identity(selection)))
            manifest.update(summary_rows=len(summary),sample_rows=len(raw),paired_rows=len(paired),receiver_diagnostic_rows=len(diagnostics),selection=selection_identity(selection))
        manifest.update(status='completed',finished=now(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:failure=exc;manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        errors=[]
        for path,digest in inputs.items():
            try:
                if sha256(path)!=digest:errors.append('Frozen input changed: '+path)
            except OSError as exc:errors.append('Unreadable frozen input: '+path+': '+str(exc))
        try:
            if manifest['training_code']!=training_code() or manifest['postprocessing_code']!=postprocessing_code_record():errors.append('Training archive or postprocessing code changed')
            validation.verify_training_source_files(manifest['training_code'])
        except Exception as exc:errors.append('Unreadable decoder code: '+str(exc))
        for kind,chain in chains.items():
            if ce_before.get(kind)!=_state_sha(chain.ce_model.state_dict()) or any(module.training for module in chain.modules()):errors.append('Frozen CE/modes changed: '+kind)
        manifest.update(frozen_files=inputs,freeze_checks=dict(passed=not errors and failure is None,errors=errors))
        if errors:failure=failure or RuntimeError(';'.join(errors));manifest.update(status='failed',error=str(failure))
        if failure is None:manifest['output_sha256']={name:sha256(output/name) for name in ARTIFACTS[stage]}
        save_json(output/'manifest.json',manifest)
    if failure is not None:raise RuntimeError('Physical-split validation failed; evidence '+str(output)) from failure
    return manifest


def verify_stage_files(output_dir,stage):
    if stage not in ARTIFACTS:raise ValueError('Unknown finish diagnostic stage')
    output=inside_root(output_dir);manifest=read_json(output/'manifest.json')
    batches=16 if stage=='calibration' else 64;start=15360 if stage=='calibration' else 16384
    if (manifest.get('status')!='completed' or manifest.get('stage')!=stage or manifest.get('namespace')!=NAMESPACE
            or manifest.get('protocol_namespace')!=cfg.NAMESPACE or manifest.get('training_code')!=training_code()
            or manifest.get('postprocessing_code')!=postprocessing_code_record()
            or manifest.get('freeze_checks',{}).get('passed') is not True or manifest.get('test_used') is not False
            or manifest.get('optimizer_updates')!=0 or manifest.get('validation_addresses')!=list(range(start,start+batches))
            or len(manifest.get('shared_inputs',[]))!=batches or manifest.get('network_forward_calls')!=batches*72
            or set(manifest.get('methods',{}))!=set(cfg.METHODS)):
        raise ValueError('Incomplete archive-bound finish stage receipt')
    acceptance=validation.load_acceptance()
    if manifest.get('training_acceptance')!=acceptance_identity(acceptance):raise ValueError('Finish training acceptance changed')
    if {name:sha256(output/name) for name in ARTIFACTS[stage]}!=manifest.get('output_sha256'):
        raise ValueError('Finish output artifact changed')
    if not manifest.get('frozen_files'):raise ValueError('Frozen input evidence is required')
    for path,digest in manifest['frozen_files'].items():
        if sha256(path)!=digest:raise ValueError('Frozen finish input changed: '+path)
    if stage=='calibration':
        if manifest.get('rzf_solves')!=batches*72*9:raise ValueError('Finish alpha grid did not reuse cached H_hat')
        chosen=select_global_alphas(manifest.get('calibration_summary',[]))
        for key,value in chosen.items():
            if canonical(manifest.get(key))!=canonical(value):raise ValueError('Calibration receipt choice differs: '+key)
    else:
        locked=load_selection();solves=batches*sum(1+int(cfg.FIXED_ALPHA!=locked['selected_alphas'][method]) for method,_,_,_ in base_conditions())
        if (manifest.get('summary_rows')!=144 or manifest.get('sample_rows')!=36864 or manifest.get('paired_rows')!=1600
                or manifest.get('selection')!=selection_identity(locked) or manifest.get('rzf_solves')!=solves
                or manifest.get('receiver_diagnostic_rows')!=199680):
            raise ValueError('Finish condition/sample/paired/receiver-diagnostic coverage changed')
    return manifest


def seal_selection(output_dir):
    """Seal completed calibration files without any additional network forward."""
    if SELECTION_PATH.exists():return load_selection()
    manifest=verify_stage_files(output_dir,'calibration')
    record=dict(format=SELECTION_FORMAT,status='passed',created=now(),grid=list(cfg.ALPHAS),
        training_code=training_code(),postprocessing_code=postprocessing_code_record(),
        training_acceptance=manifest['training_acceptance'],methods=manifest['methods'],data_protocol=manifest['data_protocol'],
        calibration_directory=str(inside_root(output_dir)),calibration_manifest_sha256=sha256(inside_root(output_dir)/'manifest.json'),
        validation_addresses=manifest['validation_addresses'],shared_inputs=manifest['shared_inputs'],
        frozen_files=manifest['frozen_files'],freeze_checks=manifest['freeze_checks'],calibration_summary=manifest['calibration_summary'],
        selected_alphas=manifest['selected_alphas'],scores=manifest['scores'],boundary_optima=manifest['boundary_optima'],test_used=False)
    record['record_id']=cfg.json_sha256(record);validate_selection_record(record);write_once_json(SELECTION_PATH,record)
    return record


def calibrate(output_dir):
    if SELECTION_PATH.exists():return load_selection()
    if (OUTPUT_ROOT/'selection_attempt.json').exists():raise RuntimeError('Partial finish alpha selection retained; no automatic reselection')
    run_stage('calibration',output_dir);return seal_selection(output_dir)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    for name in ('dry-run','calibrate','evaluate'):modes.add_argument('--'+name,action='store_true')
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(namespace=NAMESPACE,training_commit=validation.TRAINING_COMMIT,
            archived_batch=str(validation.BATCH_PATH),output_root=str(OUTPUT_ROOT),methods=cfg.METHODS,
            fixed_alpha=cfg.FIXED_ALPHA,alpha_grid=cfg.ALPHAS,calibration_addresses=[15360,15375],
            validation_addresses=[16384,16447],noise_protocol='unchanged original config',
            conditions=len(conditions()),paired_rows=len(contrasts())*8*len(METRICS),
            training=False,optimizer_created=False,weights_loaded=False,models_built=False),indent=2));return 0
    if not(args.calibrate or args.evaluate):parser.print_help();return 0
    if args.output_dir is None:parser.error('--output-dir required')
    if args.calibrate:calibrate(args.output_dir)
    else:run_stage('evaluation',args.output_dir,load_selection())
    return 0


if __name__=='__main__':raise SystemExit(main())
