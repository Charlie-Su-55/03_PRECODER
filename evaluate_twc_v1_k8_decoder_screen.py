#!/usr/bin/env python3
"""Independent frozen decoder-screen alpha calibration and paired validation."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import traceback
from configs import twc_v1_k8_decoder_screen_config as cfg
from evaluate_twc_v1_k8_rzf_probe import (score_cached_hhat,pilot_only_observation,statistics_for,
    _tensor_sha,_state_sha,_snapshot,_write_csv,METRICS,USER_RATES,DIAGNOSTICS,INVARIANT)

POLICIES={'hybrid_1_3':(1,3),'hybrid_1_1':(1,1)}
ARTIFACTS={'calibration':('alpha_calibration.csv','alpha_global_scores.csv'),
    'evaluation':('summary.csv','samples.csv.gz','paired_deltas.csv','parameters.csv','results.json',
                  'receiver_diagnostics.csv.gz','receiver_diagnostic_summary.csv')}

def _trainer():
    import train_twc_v1_k8_decoder_screen as trainer
    return trainer
def now():return datetime.now(timezone.utc).isoformat()
def source_code_record():return _trainer().source_code_record()
def sha256(path):return cfg.sha256_file(path)
def canonical(value):return cfg.normalize_json(value)
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    temporary.replace(path)
def write_once_json(path,value):
    path=cfg.inside_root(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
def method_steps(method):
    cfg.is_fdma(method)
    return (cfg.SOURCE_LOCAL_UPDATES,cfg.SOURCE_UPDATES) if method.startswith('source_') else (cfg.UPDATES,cfg.SOURCE_UPDATES+cfg.UPDATES)
def base_conditions():
    return tuple((method,1 if cfg.is_fdma(method) else 4,policy,gamma) for method in cfg.METHODS
        for policy in (('fdma',) if cfg.is_fdma(method) else tuple(POLICIES)) for gamma in cfg.GAMMAS)
def conditions():return tuple((method,mode,g,policy,gamma) for method,g,policy,gamma in base_conditions() for mode in ('fixed','tuned'))

def select_global_alphas(rows):
    expected={(method,g,policy,gamma,alpha) for method,g,policy,gamma in base_conditions() for alpha in cfg.ALPHAS};indexed={}
    for row in rows:
        key=tuple(row.get(name) for name in ('arm','group_count','energy_policy','gamma_db','alpha'));rate=row.get('sum_rate_bps_hz_mean')
        if (key not in expected or key in indexed or row.get('num_samples')!=64 or isinstance(rate,bool)
                or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<0):raise ValueError('Incomplete/duplicate/nonfinite alpha condition')
        indexed[key]=rate
    if set(indexed)!=expected:raise ValueError('Every eight-method condition and all nine alphas are required')
    selected={};scores=[];boundary={}
    for method in cfg.METHODS:
        own=[key for key in base_conditions() if key[0]==method];candidates=[]
        for alpha in cfg.ALPHAS:
            score=statistics.fmean(indexed[(*key,alpha)] for key in own)
            scores.append(dict(arm=method,alpha=alpha,mean_sum_rate=score,equal_weight_conditions=len(own)));candidates.append((score,alpha))
        alpha=min(candidates,key=lambda item:(-item[0],item[1]))[1];selected[method]=alpha;boundary[method]=alpha in (cfg.ALPHAS[0],cfg.ALPHAS[-1])
    return dict(selected_alphas=selected,scores=scores,boundary_optima=boundary)

def summarize_samples(rows,count=256):
    if type(count) is not int or count<4 or count%4:raise ValueError('Complete batches of four MU channels required')
    indexed={key:{} for key in conditions()};fingerprints={};invariant={};selected={}
    for row in rows:
        key=tuple(row.get(name) for name in ('arm','rzf_mode','group_count','energy_policy','gamma_db'));i=row.get('sample_index')
        if key not in indexed or type(i) is not int or not 0<=i<count or i in indexed[key]:raise ValueError('Unknown/duplicate complete sample key')
        step,cumulative=method_steps(key[0])
        if (row.get('seed')!=cfg.SEED or row.get('batch_index')!=i//4 or row.get('checkpoint_step')!=step
                or row.get('data_global_step')!=cumulative or row.get('source_codec_updates')!=cfg.SOURCE_UPDATES
                or row.get('new_bs_updates')!=(0 if key[0].startswith('source_') else cfg.UPDATES)
                or row.get('ue_frozen') is not True or row.get('pilot_energy_per_ue')!=16. or row.get('downlink_snr_db')!=25.):
            raise ValueError('Source/BS-only update/protocol identity mismatch')
        fp=row.get('channel_fingerprint')
        if not isinstance(fp,str) or len(fp)!=64 or any(c not in '0123456789abcdef' for c in fp) or fingerprints.setdefault(i//4,fp)!=fp:
            raise ValueError('Actual shared channel fingerprint required')
        alpha=row.get('alpha')
        if isinstance(alpha,bool) or alpha not in cfg.ALPHAS:raise ValueError('Unknown RZF alpha')
        if key[1]=='fixed' and alpha!=cfg.FIXED_ALPHA:raise ValueError('Both layouts fixed alpha must be0.03')
        if key[1]=='tuned' and selected.setdefault(key[0],alpha)!=alpha:raise ValueError('One global alpha per checkpoint required')
        for metric in METRICS+USER_RATES+DIAGNOSTICS:
            value=row.get(metric)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Invalid metric: '+metric)
        if not math.isclose(sum(row[name] for name in USER_RATES),row['sum_rate_bps_hz'],rel_tol=1e-5,abs_tol=1e-5):raise ValueError('User rates do not sum to MU rate')
        invkey=(key[0],key[2],key[3],key[4],i);values=tuple(row[name] for name in INVARIANT)
        if invariant.setdefault(invkey,values)!=values:raise ValueError('Raw H_hat NMSE/norms changed with alpha')
        if (abs(row['data_energy_per_ue']-48)>1e-4 or abs(row['frame_energy_per_ue']-64)>1e-4 or abs(row['rzf_power_mean']-1)>1e-5
                or not 0<=row.get('energy_max_abs_error',float('inf'))<=1e-4 or not 0<=row.get('rzf_power_max_abs_error',float('inf'))<=1e-5):
            raise ValueError('Energy or per-tone RZF power constraint failed')
        indexed[key][i]=row
    summary=[]
    for key,records in indexed.items():
        if set(records)!=set(range(count)):raise ValueError('Incomplete method/mode/condition/sample coverage')
        item=dict(zip(('arm','rzf_mode','group_count','energy_policy','gamma_db'),key));step,cumulative=method_steps(key[0])
        item.update(alpha=records[0]['alpha'],num_samples=count,num_batches=count//4,checkpoint_step=step,data_global_step=cumulative,
            source_codec_updates=cfg.SOURCE_UPDATES,new_bs_updates=0 if key[0].startswith('source_') else cfg.UPDATES,UE_new_updates=0,ue_frozen=True)
        for metric in METRICS+USER_RATES+DIAGNOSTICS:
            item.update({metric+'_'+name:value for name,value in statistics_for([records[i][metric] for i in range(count)]).items()})
        item.update(energy_max_abs_error=max(row['energy_max_abs_error'] for row in records.values()),rzf_power_max_abs_error=max(row['rzf_power_max_abs_error'] for row in records.values()))
        summary.append(item)
    return summary,indexed

def contrasts():
    result=[]
    for mode in ('fixed','tuned'):
        def add(name,weights):result.append(dict(name=name+'_'+mode,coefficients=weights))
        for layout,g,policies in (('fdma',1,('fdma',)),('g4',4,tuple(POLICIES))):
            for policy in policies:
                keys={v:(layout+'_'+v,mode,g,policy) for v in ('base2','deep4','dc2')};source=(cfg.source_method(layout+'_base2'),mode,g,policy)
                for left,right in (('deep4','base2'),('dc2','base2'),('dc2','deep4')):
                    add(layout+'_'+left+'_minus_'+right+'_'+policy,{keys[left]:1,keys[right]:-1})
                for v in keys:add(layout+'_'+v+'_minus_source_'+policy,{keys[v]:1,source:-1})
        for policy in POLICIES:
            f={v:('fdma_'+v,mode,1,'fdma') for v in ('base2','deep4','dc2')};g={v:('g4_'+v,mode,4,policy) for v in f}
            for v in f:add(v+'_g4_minus_fdma_'+policy,{g[v]:1,f[v]:-1})
            for left,right in (('deep4','base2'),('dc2','base2'),('dc2','deep4')):
                add('gap_interaction_'+left+'_minus_'+right+'_'+policy,{g[left]:1,f[left]:-1,g[right]:-1,f[right]:1})
    return result
def paired_statistics(rows,count=256):
    _,indexed=summarize_samples(rows,count);paired=[]
    for contrast in contrasts():
        for scope,gammas in [(str(int(g)),(g,)) for g in cfg.GAMMAS]+[('low_0_5_10',cfg.GAMMAS[:3]),('all_six',cfg.GAMMAS)]:
            for metric in METRICS:
                values=[statistics.fmean(sum(coefficient*indexed[(*key,gamma)][i][metric] for key,coefficient in contrast['coefficients'].items()) for gamma in gammas) for i in range(count)]
                batches=[statistics.fmean(values[i:i+4]) for i in range(0,count,4)]
                row=dict(contrast=contrast['name'],estimand=scope,metric=metric,num_samples=count,num_batches=count//4,
                    coefficients=json.dumps([dict(method=list(key),coefficient=value) for key,value in contrast['coefficients'].items()]),
                    ci_scope='fixed-checkpoint single-seed validation sampling; not independent training seeds')
                for prefix,v in (('sample',values),('batch',batches)):row.update({prefix+'_'+name:value for name,value in statistics_for(v).items()})
                row['batch_ci_estimable']=len(batches)>1;paired.append(row)
    return paired

def offline_diagnostics(output):
    """Truth is read only after the deployable forward has returned."""
    from models.twc_v1_k8_decoder_refinement import residual_ops
    return residual_ops.offline_decoder_diagnostics(output)
def expected_diagnostic_metrics(method):
    kinds=('all','fdma') if cfg.is_fdma(method) else ('all','key','shared')
    return tuple(f'stage_{stage}_codeword_nmse_{kind}' for stage in ('soft','round1','final') for kind in kinds)+(
        'received_residual_energy_x1','received_residual_energy_final')
def diagnostic_summary(rows,count=256):
    if type(count) is not int or count<4 or count%4:raise ValueError('Complete offline MU batches required')
    groups={(*key,metric):{} for key in base_conditions() for metric in expected_diagnostic_metrics(key[0])};fingerprints={}
    for row in rows:
        key=tuple(row[name] for name in ('arm','group_count','energy_policy','gamma_db','metric'));i=row['sample_index'];value=row['value']
        if key not in groups or type(i) is not int or not 0<=i<count or i in groups[key] or row.get('batch_index')!=i//4:
            raise ValueError('Duplicate/incomplete offline diagnostic sample identity')
        fp=row.get('channel_fingerprint')
        if not isinstance(fp,str) or len(fp)!=64 or fingerprints.setdefault(i//4,fp)!=fp:raise ValueError('Unpaired offline diagnostic channels')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Nonfinite offline diagnostic')
        groups[key][i]=value
    result=[]
    for key,values in groups.items():
        if set(values)!=set(range(count)):raise ValueError('Incomplete offline stage metric coverage')
        row=dict(zip(('arm','group_count','energy_policy','gamma_db','metric'),key));row.update(num_samples=count,num_batches=count//4)
        ordered=[values[i] for i in range(count)];batches=[statistics.fmean(ordered[i:i+4]) for i in range(0,count,4)]
        for prefix,v in (('sample',ordered),('batch',batches)):row.update({prefix+'_'+name:value for name,value in statistics_for(v).items()})
        result.append(row)
    if not result:raise ValueError('Offline receiver diagnostics are required')
    return result
def _frozen(inputs,evidence):
    for path,digest in evidence.get('frozen_files',{}).items():
        if path in inputs and inputs[path]!=digest:raise ValueError('Conflicting frozen file identity: '+path)
        inputs[path]=digest

def load_methods(torch,warmup,data_protocol,ce,acceptance):
    trainer=_trainer();sources=trainer.load_sources(torch,warmup,data_protocol,ce)
    if set(sources)!=set(cfg.SOURCE_METHODS):raise ValueError('Both accepted NMSE sources required')
    source_ue={method:trainer.tensor_digest(torch,{name.removeprefix('encoder.'):value
        for name,value in payload['codec_model'].items() if name.startswith('encoder.')})
        for method,(payload,_) in sources.items()}
    chains={};states={};methods={};parameters=[];inputs={}
    for method in cfg.METHODS:
        if method.startswith('source_'):
            payload,evidence=sources.pop(method);identity=cfg.source_identity(method,acceptance);path=Path(identity['path']);digest=identity['sha256']
            if sha256(path)!=digest:raise ValueError('Accepted source checkpoint changed')
        else:
            path=cfg.OUTPUT_ROOT/'training'/method/'checkpoint_latest.pt';payload,evidence=trainer.load_final_payload(torch,method);digest=sha256(path)
        inputs[str(path)]=digest;_frozen(inputs,evidence);kind=cfg.variant(method)
        if kind not in chains:chains[kind]=trainer.build_chain(torch,torch.device('cuda:0'),method,payload,include_stages=True,initialize=False).eval()
        chain=chains[kind];state=payload['codec_model'];expected=chain.codec_model.state_dict()
        if set(state)!=set(expected):raise ValueError('Complete trained variant state required; migration forbidden during evaluation')
        for name,value in state.items():
            if not torch.is_tensor(value) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype or not bool(torch.isfinite(value).all()):raise ValueError('Trained tensor mismatch: '+name)
        chain.codec_model.load_state_dict(state,strict=True)
        if method in cfg.ARMS:
            trainer.validate_runtime_resume(torch,chain,payload,method,acceptance,data_protocol,ce)
            if payload.get('frozen_state',{}).get('ue')!=source_ue[cfg.source_method(method)]:
                raise ValueError('Terminal checkpoint changed its frozen source UE identity')
        if trainer.tensor_digest(torch,chain.codec_model.encoder.state_dict())!=source_ue[cfg.source_method(method)]:
            raise ValueError('Actual terminal UE/condition state differs from the accepted source')
        if _state_sha(chain.codec_model.state_dict())!=_state_sha(state):
            raise ValueError('Strict terminal copy changed values or shared-decoder aliases')
        chain.eval();states[method]=state
        full=sum(p.numel() for p in chain.codec_model.parameters());ue=sum(p.numel() for p in chain.codec_model.encoder.parameters());step,cumulative=method_steps(method)
        parameter=dict(arm=method,variant=kind,codec_parameters=full,encoder_parameters=ue,bs_parameters=full-ue,
            ce_parameters=sum(p.numel() for p in chain.ce_model.parameters()),note='Actual stored numel; not equal MACs/latency')
        parameters.append(parameter);methods[method]=dict(path=str(path),sha256=digest,step=step,data_global_step=cumulative,
            source_codec_updates=cfg.SOURCE_UPDATES,new_bs_updates=0 if method.startswith('source_') else cfg.UPDATES,ue_frozen=True,
            source=cfg.source_identity(method,acceptance),strict_loading=True,migration=False,parameters=parameter,state_sha256=_state_sha(state),actual_saved_recipe=payload['recipe'])
        del payload
    return dict(chains=chains,states=states,methods=methods,parameters=parameters,frozen_files=inputs)

def validate_selection_record(record):
    identity=cfg.selection_identity(record)
    for name,value in select_global_alphas(record.get('calibration_summary',[])).items():
        if canonical(record.get(name))!=canonical(value):raise ValueError('Locked alpha choice differs: '+name)
    if (record.get('validation_addresses')!=list(range(11264,11280)) or len(record.get('shared_inputs',[]))!=16
            or record.get('freeze_checks',{}).get('passed') is not True or set(record.get('methods',{}))!=set(cfg.METHODS)
            or not record.get('source_acceptance',{}).get('record_id') or not record.get('frozen_files') or record.get('test_used') is not False):
        raise ValueError('Incomplete immutable decoder alpha evidence')
    return identity
def load_selection():
    record=read_json(cfg.SELECTION_PATH);validate_selection_record(record)
    if record['code']!=source_code_record():raise ValueError('Code changed after decoder alpha lock')
    for path,digest in record['frozen_files'].items():
        if sha256(path)!=digest:raise ValueError('Locked decoder input changed: '+path)
    directory=cfg.inside_root(record['calibration_directory'])
    if sha256(directory/'manifest.json')!=record['calibration_manifest_sha256']:raise ValueError('Alpha calibration manifest changed')
    verify_stage_files(directory,'calibration');return record

def run_stage(stage,output_dir,selection=None):
    if stage not in ARTIFACTS:raise ValueError('Unknown decoder diagnostic split')
    output=cfg.inside_root(output_dir)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',stage=stage,namespace=cfg.NAMESPACE,code=source_code_record(),test_used=False,optimizer_updates=0,
        shared_inputs=[],network_forward_calls=0,rzf_solves=0,execution={})
    inputs={};failure=None;chains={};ce_before={}
    try:
        trainer=_trainer();acceptance=trainer.load_acceptance();manifest['source_acceptance']=cfg.acceptance_identity(acceptance)
        inputs[str(cfg.SOURCE_ACCEPTANCE_PATH)]=sha256(cfg.SOURCE_ACCEPTANCE_PATH);_frozen(inputs,acceptance)
        if stage=='evaluation':
            saved=load_selection();selection=saved if selection is None else selection
            if cfg.selection_identity(selection)!=cfg.selection_identity(saved):raise ValueError('Final alpha lock differs')
            inputs[str(cfg.SELECTION_PATH)]=sha256(cfg.SELECTION_PATH)
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        import train_twc_v1_k8_codec_warmup as warmup
        device=torch.device('cuda:0');data=trainer.make_data(warmup,device);data.prepare();ce=trainer.ce_identity()
        if acceptance['code']!=source_code_record() or canonical(acceptance['data_protocol'])!=canonical(data.protocol) or acceptance['ce_identity']!=ce:
            raise ValueError('Source acceptance code/data/CE identity mismatch')
        loaded=load_methods(torch,warmup,data.protocol,ce,acceptance);chains=loaded['chains'];_frozen(inputs,loaded)
        for kind,chain in chains.items():ce_before[kind]=_state_sha(chain.ce_model.state_dict())
        for item in ce.values():inputs[item['path']]=item['sha256']
        manifest.update(data_protocol=data.protocol,ce_identity=ce,methods=loaded['methods'],parameters=loaded['parameters'],
            backend=dict(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                matmul_precision=torch.get_float32_matmul_precision(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        if stage=='evaluation':
            if selection['source_acceptance']!=cfg.acceptance_identity(acceptance) or canonical(selection['data_protocol'])!=canonical(data.protocol):raise ValueError('Locked acceptance/data mismatch')
            for path,digest in selection['frozen_files'].items():
                if inputs.get(path)!=digest:raise ValueError('Locked checkpoint/input changed: '+path)
        batches=16 if stage=='calibration' else 64;start=11264 if stage=='calibration' else 12288
        manifest['validation_addresses']=list(range(start,start+batches))
        if stage=='calibration':write_once_json(cfg.OUTPUT_ROOT/'selection_attempt.json',dict(started=now(),code=source_code_record(),grid=list(cfg.ALPHAS)))
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
                        scores=score_cached_hhat(torch,warmup._system.downlink,out['H_dl_hat'],hdl,cfg.ALPHAS if stage=='calibration' else (cfg.FIXED_ALPHA,selection['selected_alphas'][method]))
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
                                raw.extend(dict(arm=method,rzf_mode=mode,group_count=g,energy_policy=policy,gamma_db=gamma,alpha=alpha,seed=cfg.SEED,pilot_energy_per_ue=16.,downlink_snr_db=25.,sample_index=bi*4+i,batch_index=bi,channel_fingerprint=sample['sample_fingerprint'],checkpoint_step=step,data_global_step=cumulative,source_codec_updates=cfg.SOURCE_UPDATES,new_bs_updates=0 if method.startswith('source_') else cfg.UPDATES,ue_frozen=True,**{name:value[i] for name,value in arrays.items()}) for i in range(4))
                        del out,scores,ce_inputs
                    if before!=_snapshot(chain) or _state_sha(chain.codec_model.state_dict())!=loaded['methods'][method]['state_sha256']:raise ValueError('Frozen receiver state/buffers/modes changed')
                if originals!=(_tensor_sha(hdl),_tensor_sha(hul)):raise ValueError('Shared channel tensors changed')
                for gamma,pair in noises.items():
                    for kind,value in zip(('pilot','data'),pair):
                        if _tensor_sha(value)!=shared['noise'][f'{gamma}_{kind}']:raise ValueError('Shared standard noise changed')
                manifest['shared_inputs'].append(shared);del refs,noises
                print(f'Decoder {stage} batch {bi+1}/{batches}',flush=True)
        if stage=='calibration':
            summary=[dict(zip(('arm','group_count','energy_policy','gamma_db','alpha'),key),num_samples=value['count'],sum_rate_bps_hz_mean=value['rate']/value['count']) for key,value in totals.items()]
            chosen=select_global_alphas(summary);_write_csv(output/'alpha_calibration.csv',summary);_write_csv(output/'alpha_global_scores.csv',chosen['scores']);manifest.update(calibration_summary=summary,**chosen)
        else:
            summary,_=summarize_samples(raw);paired=paired_statistics(raw);secondary=diagnostic_summary(diagnostics)
            _write_csv(output/'summary.csv',summary);_write_csv(output/'samples.csv.gz',raw,True);_write_csv(output/'paired_deltas.csv',paired);_write_csv(output/'parameters.csv',loaded['parameters'])
            _write_csv(output/'receiver_diagnostics.csv.gz',diagnostics,True);_write_csv(output/'receiver_diagnostic_summary.csv',secondary)
            save_json(output/'results.json',dict(summary=summary,paired=paired,parameters=loaded['parameters'],methods=loaded['methods'],receiver_diagnostics=secondary,selection=cfg.selection_identity(selection)))
            manifest.update(summary_rows=len(summary),sample_rows=len(raw),paired_rows=len(paired),receiver_diagnostic_rows=len(diagnostics),selection=cfg.selection_identity(selection))
        manifest.update(status='completed',finished=now(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:failure=exc;manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        errors=[]
        for path,digest in inputs.items():
            try:
                if sha256(path)!=digest:errors.append('Frozen input changed: '+path)
            except OSError as exc:errors.append('Unreadable frozen input: '+path+': '+str(exc))
        try:
            if manifest['code']!=source_code_record():errors.append('Decoder code changed')
        except Exception as exc:errors.append('Unreadable decoder code: '+str(exc))
        for kind,chain in chains.items():
            if ce_before.get(kind)!=_state_sha(chain.ce_model.state_dict()) or any(module.training for module in chain.modules()):errors.append('Frozen CE/modes changed: '+kind)
        manifest.update(frozen_files=inputs,freeze_checks=dict(passed=not errors and failure is None,errors=errors))
        if errors:failure=failure or RuntimeError(';'.join(errors));manifest.update(status='failed',error=str(failure))
        if failure is None:manifest['output_sha256']={name:sha256(output/name) for name in ARTIFACTS[stage]}
        save_json(output/'manifest.json',manifest)
    if failure is not None:raise RuntimeError('Decoder validation failed; evidence '+str(output)) from failure
    return manifest

def verify_stage_files(output_dir,stage):
    output=cfg.inside_root(output_dir);manifest=read_json(output/'manifest.json');batches=16 if stage=='calibration' else 64;start=11264 if stage=='calibration' else 12288
    if (stage not in ARTIFACTS or manifest.get('status')!='completed' or manifest.get('stage')!=stage or manifest.get('namespace')!=cfg.NAMESPACE
            or manifest.get('code')!=source_code_record() or manifest.get('freeze_checks',{}).get('passed') is not True or manifest.get('test_used') is not False
            or manifest.get('optimizer_updates')!=0 or manifest.get('validation_addresses')!=list(range(start,start+batches))
            or len(manifest.get('shared_inputs',[]))!=batches or manifest.get('network_forward_calls')!=batches*72 or set(manifest.get('methods',{}))!=set(cfg.METHODS)):
        raise ValueError('Incomplete decoder evaluation receipt')
    if {name:sha256(output/name) for name in ARTIFACTS[stage]}!=manifest.get('output_sha256'):raise ValueError('Decoder artifact changed')
    for path,digest in manifest.get('frozen_files',{}).items():
        if sha256(path)!=digest:raise ValueError('Frozen decoder input changed: '+path)
    if stage=='calibration':
        if manifest.get('rzf_solves')!=batches*72*9:raise ValueError('Alpha grid did not reuse cached H_hat')
        select_global_alphas(manifest['calibration_summary'])
    else:
        locked=load_selection();solves=batches*sum(1+int(cfg.FIXED_ALPHA!=locked['selected_alphas'][method]) for method,_,_,_ in base_conditions())
        if (manifest.get('summary_rows')!=144 or manifest.get('sample_rows')!=36864 or manifest.get('paired_rows')!=1920
                or manifest.get('selection')!=cfg.selection_identity(locked) or manifest.get('rzf_solves')!=solves or manifest.get('receiver_diagnostic_rows')!=184320):
            raise ValueError('Final decoder condition/sample/paired/diagnostic coverage mismatch')
    return manifest
def seal_selection(output_dir):
    if cfg.SELECTION_PATH.exists():return load_selection()
    manifest=verify_stage_files(output_dir,'calibration')
    record=dict(format=cfg.SELECTION_FORMAT,status='passed',created=now(),grid=list(cfg.ALPHAS),code=source_code_record(),source_acceptance=manifest['source_acceptance'],methods=manifest['methods'],data_protocol=manifest['data_protocol'],calibration_directory=str(cfg.inside_root(output_dir)),calibration_manifest_sha256=sha256(cfg.inside_root(output_dir)/'manifest.json'),validation_addresses=manifest['validation_addresses'],shared_inputs=manifest['shared_inputs'],frozen_files=manifest['frozen_files'],freeze_checks=manifest['freeze_checks'],calibration_summary=manifest['calibration_summary'],selected_alphas=manifest['selected_alphas'],scores=manifest['scores'],boundary_optima=manifest['boundary_optima'],test_used=False)
    record['record_id']=cfg.json_sha256(record);validate_selection_record(record);write_once_json(cfg.SELECTION_PATH,record);return record
def calibrate(output_dir):
    if cfg.SELECTION_PATH.exists():return load_selection()
    if (cfg.OUTPUT_ROOT/'selection_attempt.json').exists():raise RuntimeError('Partial alpha selection retained; no silent reselection')
    run_stage('calibration',output_dir);return seal_selection(output_dir)
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    for name in ('dry-run','calibrate','evaluate'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(namespace=cfg.NAMESPACE,methods=cfg.METHODS,fixed_alpha=.03,base_conditions=len(base_conditions()),conditions=len(conditions()),paired_rows=len(contrasts())*8*len(METRICS),source_SHA='pending server acceptance; never guessed',alpha_calibration=[11264,11279],validation=[12288,12351],models_built=False,weights_loaded=False,test_used=False),indent=2));return 0
    if not(args.calibrate or args.evaluate):parser.print_help();return 0
    if args.output_dir is None:parser.error('--output-dir required')
    if args.calibrate:calibrate(args.output_dir)
    else:run_stage('evaluation',args.output_dir,load_selection())
    return 0
if __name__=='__main__':raise SystemExit(main())
