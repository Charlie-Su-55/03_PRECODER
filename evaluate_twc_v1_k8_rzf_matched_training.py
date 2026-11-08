#!/usr/bin/env python3
"""Lock final RZF alphas and validate six matched-training checkpoints without updates."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import traceback

from configs import twc_v1_k8_rzf_matched_training_config as cfg
from evaluate_twc_v1_k8_rzf_probe import (
    score_cached_hhat,pilot_only_observation,statistics_for,_tensor_sha,_state_sha,_snapshot,_write_csv,
    METRICS,USER_RATES,DIAGNOSTICS,INVARIANT)

POLICIES={'hybrid_1_3':(1,3),'hybrid_1_1':(1,1)}
ARTIFACTS={'calibration':('alpha_calibration.csv','alpha_global_scores.csv'),
    'evaluation':('summary.csv','samples.csv.gz','paired_deltas.csv','parameters.csv','results.json')}


def _trainer():
    import train_twc_v1_k8_rzf_matched_training as trainer
    return trainer


def now():return datetime.now(timezone.utc).isoformat()
def canonical(value):return cfg.normalize_json(value)
def sha256(path):return cfg.sha256_file(path)
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def source_code_record():return _trainer().source_code_record()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    temporary.replace(path)


def write_once_json(path,value):
    path=cfg.inside_root(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def method_steps(method):
    cfg.is_fdma(method)
    return ((cfg.SOURCE_LOCAL_UPDATES,cfg.SOURCE_UPDATES) if method.startswith('source_')
        else (cfg.UPDATES,cfg.SOURCE_UPDATES+cfg.UPDATES))


def base_conditions():
    return tuple((method,1 if cfg.is_fdma(method) else 4,policy,gamma) for method in cfg.METHODS
        for policy in (('fdma',) if cfg.is_fdma(method) else tuple(POLICIES)) for gamma in cfg.GAMMAS)


def conditions():
    return tuple((method,mode,g,policy,gamma) for method,g,policy,gamma in base_conditions() for mode in ('fixed','tuned'))


def select_global_alphas(rows):
    expected={(method,g,policy,gamma,alpha) for method,g,policy,gamma in base_conditions() for alpha in cfg.ALPHAS}
    indexed={}
    for row in rows:
        key=tuple(row.get(name) for name in ('arm','group_count','energy_policy','gamma_db','alpha'))
        rate=row.get('sum_rate_bps_hz_mean')
        if (key not in expected or key in indexed or row.get('num_samples')!=64 or isinstance(rate,bool)
                or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<0):
            raise ValueError('Incomplete, duplicate or nonfinite matched alpha calibration condition')
        indexed[key]=rate
    if set(indexed)!=expected:raise ValueError('All six methods, energy policies, gammas and nine alphas are required')
    selected={};scores=[];boundary={}
    for method in cfg.METHODS:
        own=[key for key in base_conditions() if key[0]==method];candidates=[]
        for alpha in cfg.ALPHAS:
            score=statistics.fmean(indexed[(*key,alpha)] for key in own)
            scores.append(dict(arm=method,alpha=alpha,mean_sum_rate=score,equal_weight_conditions=len(own)))
            candidates.append((score,alpha))
        value=min(candidates,key=lambda item:(-item[0],item[1]))[1]
        selected[method]=value;boundary[method]=value in (cfg.ALPHAS[0],cfg.ALPHAS[-1])
    return dict(selected_alphas=selected,scores=scores,boundary_optima=boundary)


def summarize_samples(rows,count=256):
    if type(count) is not int or count<4 or count%4:raise ValueError('Complete batches of four MU samples required')
    indexed={key:{} for key in conditions()};fingerprints={};invariant={};selected={}
    for row in rows:
        key=tuple(row.get(name) for name in ('arm','rzf_mode','group_count','energy_policy','gamma_db'))
        i=row.get('sample_index')
        if key not in indexed or type(i) is not int or not 0<=i<count or i in indexed[key]:
            raise ValueError('Missing, duplicate or unknown matched diagnostic sample key')
        step,cumulative=method_steps(key[0])
        if (row.get('seed')!=cfg.SEED or row.get('batch_index')!=i//4 or row.get('checkpoint_step')!=step
                or row.get('cumulative_updates')!=cumulative or row.get('pilot_energy_per_ue')!=16.
                or row.get('downlink_snr_db')!=25.):raise ValueError('Diagnostic source/update/protocol identity changed')
        fp=row.get('channel_fingerprint')
        if (not isinstance(fp,str) or len(fp)!=64 or any(c not in '0123456789abcdef' for c in fp)
                or fingerprints.setdefault(i//4,fp)!=fp):raise ValueError('Actual shared input fingerprint required')
        alpha=row.get('alpha')
        if isinstance(alpha,bool) or alpha not in cfg.ALPHAS:raise ValueError('Unknown diagnostic alpha')
        if key[1]=='fixed' and alpha!=cfg.training_alpha(key[0]):raise ValueError('Fixed mode must use the matched layout alpha')
        if key[1]=='tuned' and selected.setdefault(key[0],alpha)!=alpha:raise ValueError('Per-gamma or per-sample alpha selection forbidden')
        for metric in METRICS+USER_RATES+DIAGNOSTICS:
            value=row.get(metric)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid matched validation metric: '+metric)
        if not math.isclose(sum(row[name] for name in USER_RATES),row['sum_rate_bps_hz'],rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError('Per-user rates do not sum to the complete MU sum rate')
        invkey=(key[0],key[2],key[3],key[4],i);values=tuple(row[name] for name in INVARIANT)
        if invariant.setdefault(invkey,values)!=values:raise ValueError('Raw H_hat NMSE/norm changed with RZF alpha')
        if (abs(row['data_energy_per_ue']-48)>1e-4 or abs(row['frame_energy_per_ue']-64)>1e-4
                or abs(row['rzf_power_mean']-1)>1e-5 or not 0<=row.get('energy_max_abs_error',float('inf'))<=1e-4
                or not 0<=row.get('rzf_power_max_abs_error',float('inf'))<=1e-5):
            raise ValueError('Energy or per-subcarrier RZF power constraint failed')
        indexed[key][i]=row
    summary=[]
    for key,records in indexed.items():
        if set(records)!=set(range(count)):raise ValueError('Incomplete matched method/mode/condition/sample coverage')
        item=dict(zip(('arm','rzf_mode','group_count','energy_policy','gamma_db'),key));step,cumulative=method_steps(key[0])
        item.update(alpha=records[0]['alpha'],num_samples=count,num_batches=count//4,checkpoint_step=step,
            cumulative_updates=cumulative,role='validation',test_used=False)
        for metric in METRICS+USER_RATES+DIAGNOSTICS:
            item.update({metric+'_'+name:value for name,value in statistics_for([records[i][metric] for i in range(count)]).items()})
        item.update(energy_max_abs_error=max(record['energy_max_abs_error'] for record in records.values()),
            rzf_power_max_abs_error=max(record['rzf_power_max_abs_error'] for record in records.values()))
        summary.append(item)
    return summary,indexed


def contrasts():
    """Fifteen predeclared comparisons in each mode; coefficients act on raw rates."""
    result=[]
    for mode in ('fixed','tuned'):
        fj=('fdma_joint',mode,1,'fdma');fn=('fdma_nmse',mode,1,'fdma');fs=('source_fdma',mode,1,'fdma')
        for name,weights in (('fdma_nmse_minus_source_fdma',{fn:1,fs:-1}),
                ('fdma_joint_minus_source_fdma',{fj:1,fs:-1}),('fdma_joint_minus_fdma_nmse',{fj:1,fn:-1})):
            result.append(dict(name=name+'_'+mode,coefficients=weights))
        for policy in POLICIES:
            gj=('g4_joint',mode,4,policy);gn=('g4_nmse',mode,4,policy);gs=('source_g4',mode,4,policy)
            for name,weights in (('g4_nmse_minus_source_g4',{gn:1,gs:-1}),('g4_joint_minus_source_g4',{gj:1,gs:-1}),
                    ('g4_joint_minus_g4_nmse',{gj:1,gn:-1}),('g4_nmse_minus_fdma_nmse',{gn:1,fn:-1}),
                    ('g4_joint_minus_fdma_joint',{gj:1,fj:-1}),('gap_interaction',{gj:1,fj:-1,gn:-1,fn:1})):
                result.append(dict(name=name+'_'+mode+'_'+policy,coefficients=weights))
    return result


def paired_statistics(rows,count=256):
    _,indexed=summarize_samples(rows,count);paired=[]
    for contrast in contrasts():
        for scope,gammas in [(str(int(g)),(g,)) for g in cfg.GAMMAS]+[('low_0_5_10',cfg.GAMMAS[:3]),('all_six',cfg.GAMMAS)]:
            for metric in METRICS:
                values=[statistics.fmean(sum(coefficient*indexed[(*key,gamma)][i][metric]
                    for key,coefficient in contrast['coefficients'].items()) for gamma in gammas) for i in range(count)]
                batches=[statistics.fmean(values[i:i+4]) for i in range(0,count,4)]
                row=dict(contrast=contrast['name'],estimand=scope,metric=metric,num_samples=count,num_batches=count//4,
                    coefficients=json.dumps([dict(method=list(key),coefficient=value) for key,value in contrast['coefficients'].items()]),
                    sample_unit='one complete K8 MU channel',batch_unit='sample_index//4',
                    ci_scope='fixed checkpoints, single training seed, validation sampling; not multi-seed or formal test evidence')
                for prefix,values_for_stat in (('sample',values),('batch',batches)):
                    row.update({prefix+'_'+name:value for name,value in statistics_for(values_for_stat).items()})
                row['batch_ci_estimable']=len(batches)>1;paired.append(row)
    return paired


def validate_checkpoint_paths(checkpoints):
    if set(checkpoints)!=set(cfg.ARMS):raise ValueError('All four explicit matched terminal paths required')
    paths={}
    for arm,path in checkpoints.items():
        path=cfg.inside_root(path)
        if path!=(cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt').resolve():
            raise ValueError('Only this matched-training arm terminal checkpoint is allowed: '+arm)
        paths[arm]=path
    return paths


def _frozen(inputs,evidence):
    if not isinstance(evidence,dict):raise ValueError('Loader evidence mapping required')
    for path,digest in evidence.get('frozen_files',{}).items():
        if path in inputs and inputs[path]!=digest:raise ValueError('Conflicting source identity: '+path)
        inputs[path]=digest


def load_methods(torch,warmup,data_protocol,ce,calibration,checkpoints):
    trainer=_trainer();sources=trainer.load_sources(torch,warmup,data_protocol,ce)
    if set(sources)!={'source_fdma','source_g4'}:raise ValueError('Both original NMSE-only sources are required')
    states={};methods={};inputs={};parameters=[];chain=None
    for method in cfg.METHODS:
        if method.startswith('source_'):
            payload,evidence=sources.pop(method);identity=cfg.source_identity(method);path=Path(identity['path']);digest=identity['sha256']
            if sha256(path)!=digest:raise ValueError('Original NMSE checkpoint hash changed')
        else:
            path=checkpoints[method]
            payload,evidence=trainer.load_final_payload(torch,method,path,calibration,data_protocol,ce);digest=sha256(path)
        inputs[str(path)]=digest;_frozen(inputs,evidence)
        if chain is None:chain=trainer.build_chain(torch,torch.device('cuda:0'),method,payload).eval()
        expected=chain.codec_model.state_dict();state=payload['codec_model']
        if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError('Complete trained ordered state required; migration forbidden')
        for name,value in state.items():
            if (not torch.is_tensor(value) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype
                    or not bool(torch.isfinite(value).all())):raise ValueError('Final shape/dtype/finite mismatch: '+name)
        chain.codec_model.load_state_dict(state,strict=True);chain.eval()
        if any(module.training for module in chain.modules()):raise ValueError('All frozen modules must remain eval')
        states[method]=state;step,cumulative=method_steps(method)
        full=sum(parameter.numel() for parameter in chain.codec_model.parameters())
        ue=sum(parameter.numel() for parameter in chain.codec_model.encoder.parameters())
        parameter=dict(arm=method,codec_parameters=full,encoder_parameters=ue,bs_parameters=full-ue,
            ce_parameters=sum(parameter.numel() for parameter in chain.ce_model.parameters()),
            note='Actual stored numel; not MACs or latency')
        parameters.append(parameter)
        methods[method]=dict(path=str(path),sha256=digest,step=step,cumulative_updates=cumulative,
            fixed_evaluation_alpha=cfg.training_alpha(method),source=cfg.source_identity(method),
            source_implementation_commit=cfg.SOURCE_COMMIT,diagnostic_commit=cfg.DIAGNOSTIC_COMMIT,
            actual_saved_recipe=payload['recipe'],strict_loading=True,migration=False,residuals_reinitialized=False,
            state_sha256=_state_sha(state),trained_residual_sha256={name:_tensor_sha(state[name]) for name in
                ('encoder.key_window_residual.weight','encoder.shared_window_residual.weight')},parameters=parameter,loading_evidence=evidence)
        del payload
    if sources:raise ValueError('Unexpected unused source payloads')
    if len({(row['codec_parameters'],row['encoder_parameters'],row['bs_parameters']) for row in parameters})!=1:
        raise ValueError('All six matched methods must retain identical stored architecture')
    return dict(chain=chain,states=states,methods=methods,parameters=parameters,frozen_files=inputs)


def validate_selection_record(record):
    identity=cfg.selection_identity(record)
    for name,value in select_global_alphas(record.get('calibration_summary',[])).items():
        if canonical(record.get(name))!=canonical(value):raise ValueError('Locked final selection '+name+' changed')
    expected_addresses=list(range(cfg.ALPHA_CALIBRATION_START_INDEX,cfg.ALPHA_CALIBRATION_START_INDEX+cfg.ALPHA_CALIBRATION_BATCHES))
    if (record.get('validation_addresses')!=expected_addresses or len(record.get('shared_inputs',[]))!=16
            or record.get('freeze_checks',{}).get('passed') is not True or not record.get('code',{}).get('sha256')
            or not record.get('frozen_files') or set(record.get('methods',{}))!=set(cfg.METHODS)
            or not record.get('beta_calibration',{}).get('record_id') or record.get('test_used') is not False):
        raise ValueError('Incomplete independent final-alpha source/calibration/freeze evidence')
    return identity


def load_selection():
    record=read_json(cfg.SELECTION_PATH);validate_selection_record(record)
    if record['code']!=source_code_record():raise ValueError('Matched code changed after final-alpha lock')
    for path,digest in record['frozen_files'].items():
        if sha256(path)!=digest:raise ValueError('Locked final-alpha input changed: '+path)
    directory=cfg.inside_root(record['calibration_directory'])
    if sha256(directory/'manifest.json')!=record.get('calibration_manifest_sha256'):
        raise ValueError('Final-alpha calibration manifest changed')
    verify_stage_files(directory,'calibration')
    return record


def run_stage(stage,output_dir,selection=None,checkpoints=None):
    if stage not in ('calibration','evaluation'):raise ValueError('Unknown matched final diagnostic stage')
    if checkpoints is None:checkpoints={arm:cfg.OUTPUT_ROOT/'training'/arm/'checkpoint_latest.pt' for arm in cfg.ARMS}
    checkpoints=validate_checkpoint_paths(checkpoints)
    output=cfg.inside_root(output_dir)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',stage=stage,code=source_code_record(),test_used=False,optimizer_updates=0,
        namespace=cfg.NAMESPACE,source_implementation_commit=cfg.SOURCE_COMMIT,diagnostic_commit=cfg.DIAGNOSTIC_COMMIT,
        shared_inputs=[],network_forward_calls=0,rzf_solves=0)
    inputs={};failure=None;chain=None;ce_before=None
    try:
        trainer=_trainer();calibration=trainer.load_calibration()
        inputs[str(cfg.CALIBRATION_PATH)]=sha256(cfg.CALIBRATION_PATH);_frozen(inputs,calibration)
        if stage=='evaluation':
            locked=load_selection()
            if selection is None:selection=locked
            if cfg.selection_identity(selection)!=cfg.selection_identity(locked):raise ValueError('Explicit final alpha lock differs from saved record')
            inputs[str(cfg.SELECTION_PATH)]=sha256(cfg.SELECTION_PATH)
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        import train_twc_v1_k8_codec_warmup as warmup
        device=torch.device('cuda:0');data=trainer.make_data(warmup,device);data.prepare();ce=trainer.ce_identity()
        if (calibration['code']!=source_code_record() or canonical(calibration['data_protocol'])!=canonical(data.protocol)
                or calibration['ce_identity']!=ce):raise ValueError('New beta calibration code/data/CE mismatch')
        loaded=load_methods(torch,warmup,data.protocol,ce,calibration,checkpoints)
        chain=loaded['chain'];states=loaded['states'];_frozen(inputs,loaded)
        for record in ce.values():inputs[record['path']]=record['sha256']
        ce_before=_state_sha(chain.ce_model.state_dict())
        manifest.update(data_protocol=data.protocol,ce_identity=ce,methods=loaded['methods'],parameters=loaded['parameters'],
            beta_calibration=cfg.calibration_identity(calibration),backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0),matmul_precision=torch.get_float32_matmul_precision(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        if stage=='evaluation':
            if canonical(selection['data_protocol'])!=canonical(data.protocol) or selection['beta_calibration']!=cfg.calibration_identity(calibration):
                raise ValueError('Final-alpha lock data/new beta identity changed')
            for path,digest in selection['frozen_files'].items():
                if inputs.get(path)!=digest:raise ValueError('Final-alpha checkpoint/input identity changed: '+path)
        batches=cfg.ALPHA_CALIBRATION_BATCHES if stage=='calibration' else cfg.VALIDATION_BATCHES
        start=cfg.ALPHA_CALIBRATION_START_INDEX if stage=='calibration' else cfg.VALIDATION_START_INDEX
        manifest.update(validation_addresses=list(range(start,start+batches)))
        if stage=='calibration':
            write_once_json(cfg.OUTPUT_ROOT/'selection_attempt.json',dict(started=now(),code=source_code_record(),grid=list(cfg.ALPHAS)))
        raw=[];totals={};downlink=warmup._system.downlink
        with torch.no_grad():
            for bi in range(batches):
                sample=data.sample('validation',start+bi);hdl,hul=sample['H_dl'],sample['H_ul']
                originals=(_tensor_sha(hdl),_tensor_sha(hul));noises={};refs={}
                shared=dict(batch_index=bi,address=start+bi,H_dl_sha256=originals[0],H_ul_sha256=originals[1],
                    sample_fingerprint=sample['sample_fingerprint'],noise={},ce_inputs={})
                for gi,gamma in enumerate(cfg.GAMMAS):
                    pair=[]
                    for kind in ('pilot','data'):
                        noise_stage='alpha_calibration' if stage=='calibration' else 'evaluation'
                        generator=torch.Generator(device='cpu').manual_seed(cfg.noise_seed(noise_stage,bi,gi,kind))
                        noise=torch.view_as_complex(torch.randn(4,32,256,2,generator=generator)).to(device);pair.append(noise)
                        shared['noise'][f'{gamma}_{kind}']=_tensor_sha(noise)
                    noises[gamma]=tuple(pair)
                pilot_indices=sorted({q for row in warmup.protocol.PILOT_POSITIONS_BY_USER for q in row})
                known_sha=_tensor_sha(chain._known_pilots(4,device))
                for method in cfg.METHODS:
                    chain.codec_model.load_state_dict(states[method],strict=True);chain.eval();before=_snapshot(chain)
                    for _,g,policy,gamma in (condition for condition in base_conditions() if condition[0]==method):
                        if cfg.is_fdma(method):layout,split=warmup.protocol.FDMALayout(),None
                        else:
                            ratio=POLICIES[policy];layout=warmup.protocol.default_hybrid_layout(4)
                            split=warmup.protocol.EnergySplit(16.,48.*ratio[0]/sum(ratio),48.*ratio[1]/sum(ratio))
                        pilot,noise=noises[gamma]
                        out=chain(hdl,hul,feedback_gamma_db=gamma,pilot_energy_per_ue=16.,pilot_standard_noise=pilot,
                            data_standard_noise=noise,layout=layout,energy_split=split);manifest['network_forward_calls']+=1
                        ce_inputs=(pilot_only_observation(torch,out,pilot_indices),out['H_ul_ls'],out['H_ul_est'])
                        if gamma not in refs:
                            refs[gamma]=tuple(value.detach().clone() for value in ce_inputs)
                            shared['ce_inputs'][str(gamma)]=dict(pilot_y_sha256=_tensor_sha(ce_inputs[0]),known_pilots_sha256=known_sha,
                                H_ls_sha256=_tensor_sha(ce_inputs[1]),H_ul_est_sha256=_tensor_sha(ce_inputs[2]),Ep=16.,N0=out['n0'])
                        elif any(not torch.equal(left,right) for left,right in zip(refs[gamma],ce_inputs)):
                            raise ValueError('Actual CE inputs/estimate differ across matched methods or ratios')
                        alphas=cfg.ALPHAS if stage=='calibration' else (cfg.training_alpha(method),selection['selected_alphas'][method])
                        scores=score_cached_hhat(torch,downlink,out['H_dl_hat'],hdl,alphas);manifest['rzf_solves']+=len(scores)
                        energy=out['energy']['actual_data_per_ue'];energy_error=(energy-48.).abs().amax(-1)
                        if not bool(torch.isfinite(energy).all()) or energy_error.max().item()>1e-4:raise ValueError('UE energy constraint failed')
                        if stage=='calibration':
                            for alpha,score in scores.items():
                                sums=totals.setdefault((method,g,policy,gamma,alpha),dict(count=0,rate=0.))
                                sums['count']+=4;sums['rate']+=math.fsum(score['sum_rate_bps_hz'].cpu().tolist())
                        else:
                            step,cumulative=method_steps(method)
                            for mode,alpha in (('fixed',cfg.training_alpha(method)),('tuned',selection['selected_alphas'][method])):
                                values=dict(scores[alpha],data_energy_per_ue=energy.mean(-1),frame_energy_per_ue=energy.mean(-1)+16.,energy_max_abs_error=energy_error)
                                arrays={name:value.cpu().tolist() for name,value in values.items()}
                                raw.extend(dict(arm=method,rzf_mode=mode,group_count=g,energy_policy=policy,gamma_db=gamma,alpha=alpha,
                                    seed=cfg.SEED,pilot_energy_per_ue=16.,downlink_snr_db=25.,sample_index=bi*4+i,batch_index=bi,
                                    channel_fingerprint=sample['sample_fingerprint'],checkpoint_step=step,cumulative_updates=cumulative,
                                    **{name:value[i] for name,value in arrays.items()}) for i in range(4))
                        del out,scores,ce_inputs
                    if before!=_snapshot(chain) or _state_sha(chain.codec_model.state_dict())!=loaded['methods'][method]['state_sha256']:
                        raise ValueError('Frozen trained state/buffers/modes changed')
                if originals!=(_tensor_sha(hdl),_tensor_sha(hul)):raise ValueError('Shared channel tensors changed')
                for gamma,pair in noises.items():
                    for kind,value in zip(('pilot','data'),pair):
                        if _tensor_sha(value)!=shared['noise'][f'{gamma}_{kind}']:raise ValueError('Standard noise changed')
                manifest['shared_inputs'].append(shared);del refs,noises
                print(f'Matched {stage} batch {bi+1}/{batches}',flush=True)
        if stage=='calibration':
            summary=[dict(zip(('arm','group_count','energy_policy','gamma_db','alpha'),key),num_samples=value['count'],
                sum_rate_bps_hz_mean=value['rate']/value['count']) for key,value in totals.items()]
            chosen=select_global_alphas(summary);_write_csv(output/'alpha_calibration.csv',summary);_write_csv(output/'alpha_global_scores.csv',chosen['scores'])
            manifest.update(calibration_summary=summary,**chosen)
        else:
            summary,_=summarize_samples(raw);paired=paired_statistics(raw)
            _write_csv(output/'summary.csv',summary);_write_csv(output/'samples.csv.gz',raw,True)
            _write_csv(output/'paired_deltas.csv',paired);_write_csv(output/'parameters.csv',loaded['parameters'])
            save_json(output/'results.json',dict(summary=summary,paired=paired,parameters=loaded['parameters'],methods=loaded['methods'],
                selection=cfg.selection_identity(selection),beta_calibration=cfg.calibration_identity(calibration)))
            manifest.update(summary_rows=len(summary),sample_rows=len(raw),paired_rows=len(paired),selection=cfg.selection_identity(selection))
        manifest.update(status='completed',finished=now())
    except BaseException as error:
        failure=error;manifest.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        errors=[]
        for path,digest in inputs.items():
            try:
                if not Path(path).is_file() or sha256(path)!=digest:errors.append('Frozen input changed: '+path)
            except OSError as error:errors.append('Frozen input unreadable: '+path+': '+str(error))
        try:
            if manifest['code']!=source_code_record():errors.append('Matched source code changed')
        except Exception as error:errors.append('Code unreadable: '+str(error))
        if chain is not None:
            if ce_before is None or _state_sha(chain.ce_model.state_dict())!=ce_before:errors.append('Frozen CE state changed or was not initialized')
            if any(module.training for module in chain.modules()):errors.append('Eval modes changed')
        manifest.update(frozen_files=inputs,freeze_checks=dict(passed=not errors and failure is None,errors=errors))
        if errors:failure=failure or RuntimeError('; '.join(errors));manifest.update(status='failed',error=str(failure))
        if failure is None:manifest['output_sha256']={name:sha256(output/name) for name in ARTIFACTS[stage]}
        save_json(output/'manifest.json',manifest)
    if failure is not None:raise RuntimeError('Matched final validation failed; evidence '+str(output)) from failure
    return manifest


def verify_stage_files(output_dir,stage):
    if stage not in ARTIFACTS:raise ValueError('Unknown matched stage receipt')
    output=cfg.inside_root(output_dir);manifest=read_json(output/'manifest.json')
    batches=cfg.ALPHA_CALIBRATION_BATCHES if stage=='calibration' else cfg.VALIDATION_BATCHES
    start=cfg.ALPHA_CALIBRATION_START_INDEX if stage=='calibration' else cfg.VALIDATION_START_INDEX
    if (manifest.get('status')!='completed' or manifest.get('stage')!=stage or manifest.get('namespace')!=cfg.NAMESPACE
            or manifest.get('code')!=source_code_record() or manifest.get('freeze_checks',{}).get('passed') is not True
            or manifest.get('optimizer_updates')!=0 or manifest.get('test_used') is not False
            or manifest.get('validation_addresses')!=list(range(start,start+batches))
            or len(manifest.get('shared_inputs',[]))!=batches or manifest.get('network_forward_calls')!=batches*54
            or set(manifest.get('methods',{}))!=set(cfg.METHODS) or not manifest.get('frozen_files')):
        raise ValueError('Incomplete frozen matched-alpha stage receipt')
    if {name:sha256(output/name) for name in ARTIFACTS[stage]}!=manifest.get('output_sha256'):
        raise ValueError('Completed matched diagnostic artifact changed')
    for path,digest in manifest['frozen_files'].items():
        if sha256(path)!=digest:raise ValueError('Frozen matched stage input changed: '+path)
    if stage=='calibration':
        if manifest.get('rzf_solves')!=batches*54*9:raise ValueError('Calibration did not reuse each cached H_hat for all nine alphas')
        select_global_alphas(manifest.get('calibration_summary',[]))
    else:
        locked=load_selection()
        expected_solves=batches*sum(1+int(cfg.training_alpha(method)!=locked['selected_alphas'][method]) for method,_,_,_ in base_conditions())
        if (manifest.get('summary_rows')!=108 or manifest.get('sample_rows')!=27648 or manifest.get('paired_rows')!=960
                or manifest.get('selection')!=cfg.selection_identity(locked) or manifest.get('rzf_solves')!=expected_solves):
            raise ValueError('Final six-method fixed/tuned coverage, alpha lock or cache reuse mismatch')
    return manifest


def seal_selection(output_dir):
    if cfg.SELECTION_PATH.exists():return load_selection()
    manifest=verify_stage_files(output_dir,'calibration')
    record=dict(format=cfg.SELECTION_FORMAT,status='passed',created=now(),grid=list(cfg.ALPHAS),code=source_code_record(),
        beta_calibration=manifest['beta_calibration'],methods=manifest['methods'],data_protocol=manifest['data_protocol'],
        calibration_directory=str(cfg.inside_root(output_dir)),calibration_manifest_sha256=sha256(cfg.inside_root(output_dir)/'manifest.json'),
        validation_addresses=manifest['validation_addresses'],shared_inputs=manifest['shared_inputs'],
        frozen_files=manifest['frozen_files'],freeze_checks=manifest['freeze_checks'],calibration_summary=manifest['calibration_summary'],
        selected_alphas=manifest['selected_alphas'],scores=manifest['scores'],boundary_optima=manifest['boundary_optima'],test_used=False)
    record['record_id']=cfg.json_sha256(record);validate_selection_record(record);write_once_json(cfg.SELECTION_PATH,record)
    return record


def calibrate(output_dir):
    if cfg.SELECTION_PATH.exists():return load_selection()
    attempt=cfg.OUTPUT_ROOT/'selection_attempt.json'
    if attempt.exists():raise RuntimeError('Incomplete alpha-selection attempt retained; no automatic reselection')
    run_stage('calibration',output_dir);return seal_selection(output_dir)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--calibrate',action='store_true');mode.add_argument('--evaluate',action='store_true')
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE,methods=cfg.METHODS,fixed_alphas=cfg.TRAINING_ALPHAS,
            alpha_grid=cfg.ALPHAS,base_conditions=len(base_conditions()),final_conditions=len(conditions()),
            calibration_samples=64,calibration_addresses=[6144,6159],validation_samples=256,validation_addresses=[7168,7231],
            paired_rows=len(contrasts())*8*len(METRICS),weights_loaded=False,models_built=False,test_used=False),indent=2));return 0
    if not(args.calibrate or args.evaluate):parser.print_help();return 0
    if args.output_dir is None:parser.error('--output-dir required')
    if args.calibrate:calibrate(args.output_dir)
    else:run_stage('evaluation',args.output_dir,load_selection())
    return 0


if __name__=='__main__':raise SystemExit(main())
