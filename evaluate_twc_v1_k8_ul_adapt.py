#!/usr/bin/env python3
"""Paired source/fixed-terminal validation after all four adaptation runs.

The four trained terminals are accepted before a final-data factory is
constructed. Source and final models then replay the same newly addressed
inputs. Simulator truth is supplied only to offline metric functions.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time

from evaluate_twc_v1_k8_icc_aligned import _tensor_hash
from physical_split_finish_io import write_union_csv

EVALUATION_FILES = ('samples.csv.gz', 'summary.csv', 'comparisons.csv',
    'input_fingerprints.json', 'source_records.json', 'training_pairing.json')
INPUT_FIELDS = frozenset(('H_dl', 'H_ul', 'standard_noise', 'pilot_polarity', 'gamma',
    'wave_active', 'Y_rx', 'estimated_UL', 'CE_variance', 'prior'))
RECOVERY_FIELDS = frozenset(('soft', 'reliability', 'estimated_UL', 'variance'))


def _config():
    from configs import twc_v1_k8_ul_adapt_config as cfg
    return cfg


def _runtime():
    import ul_adapt_runtime as runtime
    return runtime


def _trainer():
    import train_twc_v1_k8_ul_adapt as trainer
    return trainer


def _results():
    import twc_v1_k8_ul_adapt_results as results
    return results


def checked_output(path):
    runtime,cfg=_runtime(),_config()
    path=runtime.inside(path)
    subtree=(cfg.OUTPUT_ROOT/'evaluation').resolve()
    if path==subtree or not path.is_relative_to(subtree):
        raise ValueError('Proper owned adaptation evaluation attempt required')
    return path


def _valid_hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _records_signature(codec,stats,ce):
    runtime=_runtime()
    modules=dict(codec=codec,statistics=stats,CE=ce)
    for module in modules.values():
        if (any(child.training for child in module.modules())
                or any(p.requires_grad for p in module.parameters())):
            raise ValueError('Final source/terminal networks must be frozen in evaluation mode')
    return {name:runtime.state_digest(module) for name,module in modules.items()}


def _check_files(record, *, fixture):
    runtime=_runtime()
    if not fixture:
        runtime.verify_files(record)
        return
    if not isinstance(record,dict) or not record.get('frozen_files'):
        raise ValueError('Explicit read-only CPU fixture source evidence required')
    for item in record['frozen_files']:
        path=Path(item['path'])
        if path.is_symlink() or not path.is_file() or runtime.sha256(path)!=item['sha256']:
            raise ValueError('Read-only source/terminal bytes changed')


def _check_terminal_record(record, arm, method, *, fixture):
    cfg=_config()
    if not isinstance(record,dict) or record.get('step')!=cfg.UPDATES:
        raise ValueError('All four fixed training terminals must be completed before final data')
    if record.get('arm',arm)!=arm or record.get('method',method)!=method:
        raise ValueError('Terminal physical arm or receiver method changed')
    if not fixture:
        if record.get('core')!=_runtime().core_identity():
            raise ValueError('Terminal adaptation scientific core changed')
        if record.get('production_evidence') is not True:
            raise ValueError('CPU fixture terminal cannot pass production evaluation')
    _check_files(record,fixture=fixture)


def _verify_terminal_metadata(arm,method,record):
    """Recheck accepted training files without allocating a network or GPU."""
    cfg,runtime=_config(),_runtime()
    manifest=_trainer().verify_terminal_record(arm,method)
    directory=runtime.inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm,method))
    digest=record.get('checkpoint_sha256',record.get('sha256'))
    if (manifest.get('checkpoint_sha256')!=digest
            or record.get('calibration_identity')!=cfg.calibration_identity()
            or record.get('counters')!=manifest.get('counters')):
        raise ValueError('Accepted terminal payload/recipe/calibration identity changed')
    for field,name in (('manifest_sha256','manifest.json'),
            ('terminal_receipt_sha256','terminal_receipt.json'),
            ('initialization_sha256','initialization.json')):
        if record.get(field)!=runtime.sha256(directory/name):
            raise ValueError('Accepted terminal receipt bytes changed: '+name)


def _calibration_files():
    cfg=_config()
    directory=Path(cfg.CALIBRATION_DIR)
    return directory/'model.json',directory/'manifest.json'


def _calibration_identity():
    runtime,cfg=_runtime(),_config()
    model,manifest=_calibration_files()
    for path in (model,manifest):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Pinned calibration regular files required')
    identity=dict(model_sha256=runtime.sha256(model),manifest_sha256=runtime.sha256(manifest))
    if (identity['model_sha256']!=cfg.CALIBRATION_MODEL_SHA256
            or identity['manifest_sha256']!=cfg.CALIBRATION_MANIFEST_SHA256):
        raise ValueError('Pinned calibration model or manifest changed')
    return identity


def _base_manifest(fixture,core):
    cfg=_config()
    return dict(format=cfg.NAMESPACE+'_evaluation_v1',phase='final',status='running',
        core=core,recipe_fingerprints={cfg.key(a,m):cfg.fingerprint(a,m)
            for a in cfg.ARMS for m in cfg.TRAIN_METHODS},
        phases=list(cfg.PHASES),arms=list(cfg.ARMS),methods=list(cfg.TRAIN_METHODS),
        gamma_db=list(cfg.GAMMAS),batch_size=cfg.MICRO_BATCH,n_batches=cfg.FINAL_BATCHES,
        samples_per_condition=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,
        fixed_rzf_alpha=cfg.ALPHA,DL_SNR=cfg.DL_SNR,terminal_updates=cfg.UPDATES,
        test_used=False,checkpoint_selection=False,calibrator_refitted=False,
        neural_updates_during_evaluation=0,UE_updates=0,CE_updates=0,EMA_updates=0,
        production_evidence=not fixture,source_checkpoints_verified=not fixture,
        all_terminals_verified_before_data=True)


def score_out(out,H_dl):
    """Use the unchanged actual CSI/RZF/rate scorer with optional soft calibration."""
    import torch
    from evaluate_twc_v1_k8_icc_rate import metric_vectors
    result={name:torch.as_tensor(value,device=H_dl.device,dtype=torch.float32).detach()
        for name,value in metric_vectors(out,H_dl).items()}
    if 'recovered' in out:
        from twc_v1_k8_ul_calibration_metrics import recovery_vectors
        values=recovery_vectors(out['recovered'],out['waveform'].active,out['layout'])
        for name,value in values.items():
            if name in result and not torch.allclose(result[name],value,rtol=2e-5,atol=1e-7):
                raise ValueError('Duplicate soft/rate metric definitions disagree: '+name)
            result[name]=value.detach()
    return result


def _validate_vectors(vectors,batch):
    import torch
    if not set(_results().REQUIRED_METRICS)<=set(vectors):
        raise ValueError('All final rate, CSI and soft-error vectors required')
    if any(not torch.is_tensor(value) or tuple(value.shape)!=(batch,)
            or value.requires_grad or not bool(torch.isfinite(value).all())
            for value in vectors.values()):
        raise ValueError('Finite detached complete-MU evaluation vectors required')


def _recovery_hashes(out):
    values=dict(soft=_tensor_hash(out['soft']),reliability=_tensor_hash(out['reliability']),
        estimated_UL=_tensor_hash(out['H_ul_est']))
    if 'recovered' in out:
        variance=out['recovered']['variance']
    else:
        active=out['waveform'].active.shape[-1]
        variance=out['reliability'][...,:active].reciprocal()
    values['variance']=_tensor_hash(variance)
    return values


def fingerprint_map(entries, *, batches=None, device=None):
    """Verify one common input record per arm/batch/gamma with all four views."""
    import torch
    cfg,runtime=_config(),_runtime()
    batches=cfg.FINAL_BATCHES if batches is None else batches
    expected={(arm,bi,float(g)) for arm in cfg.ARMS for bi in range(batches) for g in cfg.GAMMAS}
    conditions={p+'_'+m for p in cfg.PHASES for m in cfg.TRAIN_METHODS}
    index={}
    metadata=None
    for row in entries:
        key=row.get('arm'),row.get('batch_index'),row.get('gamma_db')
        if key not in expected or key in index or row.get('role')!='final':
            raise ValueError('Duplicate/unexpected final paired input condition')
        if device is None: device=row.get('device')
        if metadata is None: metadata=runtime.make_data(device)
        if (row.get('device')!=str(device)
                or row.get('address')!=metadata.address('validation',cfg.ROLE_RANGES['final'][0]+key[1])
                or set(row.get('input_hashes',{}))!=INPUT_FIELDS
                or any(not _valid_hash(v) for v in row['input_hashes'].values())
                or set(row.get('conditions',[]))!=conditions
                or len(row.get('conditions',[]))!=len(conditions)):
            raise ValueError('Incomplete final input identity/address/condition coverage')
        expected_gamma=_tensor_hash(torch.full((cfg.MICRO_BATCH,),key[2],dtype=torch.float32))
        if row['input_hashes']['gamma']!=expected_gamma:
            raise ValueError('Actual gamma tensor differs from declared final point')
        recoveries=row.get('recovery_hashes',{})
        if (set(recoveries)!=set(cfg.TRAIN_METHODS)
                or any(set(r)!=RECOVERY_FIELDS or any(not _valid_hash(v) for v in r.values())
                    for r in recoveries.values())
                or set(row.get('decoder_output_hashes',{}))!=conditions
                or any(not _valid_hash(v) for v in row['decoder_output_hashes'].values())):
            raise ValueError('Complete decoder-independent recovery and output hashes required')
        index[key]=row
    if set(index)!=expected:
        raise ValueError('Missing paired final input condition')
    common=('H_dl','H_ul','standard_noise','pilot_polarity','gamma','estimated_UL','CE_variance')
    for bi in range(batches):
        for gamma in cfg.GAMMAS:
            pair=[index[arm,bi,float(gamma)]['input_hashes'] for arm in cfg.ARMS]
            if any(any(item[name]!=pair[0][name] for name in common) for item in pair[1:]):
                raise ValueError('Physical arms did not use paired channel/CE/noise/pilots')
    for arm in cfg.ARMS:
        for field in ('H_dl','H_ul'):
            values=[]
            for bi in range(batches):
                repeated={index[arm,bi,float(g)]['input_hashes'][field] for g in cfg.GAMMAS}
                if len(repeated)!=1:
                    raise ValueError('Gamma views must retain the same complete-MU channel sample')
                values.append(next(iter(repeated)))
            if len(set(values))!=batches:
                raise ValueError('Final replay repeated channel payloads across distinct batch addresses')
    return index


def _add_fingerprint(index,phase,arm,method,bi,sample,out,device):
    cfg=_config()
    common=dict(out['input_hashes'],H_dl=_tensor_hash(sample['H_dl']),H_ul=_tensor_hash(sample['H_ul']))
    if set(common)!=INPUT_FIELDS:
        raise ValueError('Runtime omitted actual input hashes')
    gamma=float(out['gamma_db'][0])
    key=arm,bi,gamma
    condition=phase+'_'+method
    recovered=_recovery_hashes(out)
    if key not in index:
        index[key]=dict(role='final',arm=arm,batch_index=bi,gamma_db=gamma,
            address=sample['address'],device=str(device),input_hashes=common,
            conditions=[],recovery_hashes={},decoder_output_hashes={})
    entry=index[key]
    if (entry['input_hashes']!=common or entry['address']!=sample['address']
            or condition in entry['conditions']):
        raise ValueError('Source/final models did not replay identical newly addressed inputs')
    if method in entry['recovery_hashes'] and entry['recovery_hashes'][method]!=recovered:
        raise ValueError('Decoder adaptation changed frozen receiver outputs')
    entry['conditions'].append(condition)
    entry['recovery_hashes'][method]=recovered
    entry['decoder_output_hashes'][condition]=_tensor_hash(out['H_hat'])


def _check_calibration_disjoint(entries):
    cfg,runtime=_config(),_runtime()
    path=Path(cfg.CALIBRATION_DIR)/'input_fingerprints.json'
    calibration=runtime.read_json(path)
    for field in ('H_dl','H_ul'):
        previous={r['fingerprint'][field] for r in calibration}
        if previous.intersection(r['input_hashes'][field] for r in entries):
            raise ValueError('Final channels overlap the locked calibration labels')


def _write_gzip(path,rows):
    fields=sorted(set().union(*(set(row) for row in rows)))
    with gzip.open(path,'wt',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _unique_json_pairs(items):
    value={}
    for key,item in items:
        if key in value:raise ValueError('Duplicate training-journal JSON field')
        value[key]=item
    return value


def verify_training_pairing(terminal_records, *, fixture=False):
    """Check saved-boundary snapshots and the four full-input trace endpoints.

    Snapshot payloads independently verify the final microbatches of each
    saved update. The rolling digest covers all actual input records emitted
    by training, but is not an independent replay of omitted channel tensors.
    A failed save can leave an uncommitted journal row: the last complete row
    per boundary is canonical, while all historical rows remain on disk.
    """
    import torch
    cfg,runtime,trainer=_config(),_runtime(),_trainer()
    if fixture:
        return dict(format=cfg.NAMESPACE+'_training_pairing_v1',status='test_fixture_skipped',
            production_evidence=False,full_micro_trace_chain_verified=False,
            snapshot_pairing_verified=False,hashed_microbatches=0,
            logged_boundary_updates=0,logged_microbatches=0,trajectories={})
    expected_keys={cfg.key(a,m) for a in cfg.ARMS for m in cfg.TRAIN_METHODS}
    if set(terminal_records)!=expected_keys:
        raise ValueError('Four training trajectories required for paired acceptance')
    expected_steps=set(range(cfg.SAVE_EVERY,cfg.UPDATES+1,cfg.SAVE_EVERY))
    records,selected,protocol={}, {},None
    for arm in cfg.ARMS:
        for method in cfg.TRAIN_METHODS:
            key=cfg.key(arm,method)
            terminal=terminal_records[key]
            current=terminal.get('data_protocol')
            if not isinstance(current,dict) or current.get('execution',{}).get('device')!='cuda:0':
                raise ValueError('Canonical CUDA training data protocol required')
            metadata=runtime.make_data('cuda:0')
            if current!=metadata.protocol or protocol is not None and current!=protocol:
                raise ValueError('Training trajectories did not use one paired replay protocol')
            protocol=current
            directory=runtime.inside(cfg.OUTPUT_ROOT/'training'/key)
            path=runtime.inside(directory/'progress.jsonl')
            if (path.is_symlink() or not path.is_file() or path.stat().st_size>64*1024**2
                    or runtime.sha256(path)!=terminal.get('progress_sha256')):
                raise ValueError('Complete immutable training progress journal required')
            accepted={};count=0;previous_step=0
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    if len(line)>1024**2:raise ValueError('Training journal row exceeds bounded schema')
                    row=json.loads(line,object_pairs_hook=_unique_json_pairs)
                    step=row.get('step')
                    if (type(step) is not int or step not in expected_steps or step<previous_step
                            or row.get('arm')!=arm or row.get('method')!=method
                            or row.get('counters')!=cfg.counters(step)):
                        raise ValueError('Training saved-boundary identity/count changed')
                    previous_step=step
                    items=row.get('input_fingerprints')
                    if not isinstance(items,list) or len(items)!=cfg.ACCUMULATION:
                        raise ValueError('Each saved training boundary requires every effective-batch micro')
                    trace=trainer.validate_training_trace(row.get('training_trace_before_update'),
                        protocol,microbatches=(step-1)*cfg.ACCUMULATION)
                    for micro,item in enumerate(items):
                        index=cfg.micro_index(step,micro)
                        fields=(set(INPUT_FIELDS)-{'H_dl','H_ul'})|{'channel'}
                        if (not isinstance(item,dict) or item.get('micro_index')!=index
                                or item.get('address')!=metadata.address('codec_train',index)
                                or not fields <= set(item)):
                            raise ValueError('Actual logged training address or microbatch changed')
                        if any(not _valid_hash(item.get(name)) for name in fields):
                            raise ValueError('Actual logged training tensor SHA256 required')
                        gamma=_tensor_hash(torch.tensor(cfg.gamma_schedule(index),dtype=torch.float32))
                        if item['gamma']!=gamma:
                            raise ValueError('Logged actual training gamma differs from the fixed balanced schedule')
                        trace=trainer.advance_training_trace(trace,item,protocol)
                    if trace!=row.get('training_trace'):
                        raise ValueError('Logged actual microbatches do not close their rolling trace endpoint')
                    trainer.validate_training_trace(trace,protocol,microbatches=step*cfg.ACCUMULATION)
                    accepted[step]=row
                    count+=1
            if set(accepted)!=expected_steps:
                raise ValueError('A complete saved-boundary training journal is required before final data')
            final=trainer.validate_training_trace(terminal.get('training_trace'),protocol,
                microbatches=cfg.UPDATES*cfg.ACCUMULATION)
            if final!=accepted[cfg.UPDATES]['training_trace']:
                raise ValueError('Terminal training trace differs from its final accepted journal boundary')
            selected[key]=accepted
            records[key]=dict(progress_sha256=terminal['progress_sha256'],
                logged_boundary_updates=len(accepted),logged_microbatches=len(accepted)*cfg.ACCUMULATION,
                retained_journal_rows=count,superseded_journal_rows=count-len(accepted),
                training_trace=final,snapshot_sha256=hashlib.sha256(_canonical([
                    dict(step=step,input_fingerprints=accepted[step]['input_fingerprints'],
                        training_trace_before_update=accepted[step]['training_trace_before_update'],
                        training_trace=accepted[step]['training_trace'])
                    for step in sorted(accepted)])).hexdigest())
    traces=[row['training_trace'] for row in records.values()]
    if any(trace!=traces[0] for trace in traces[1:]):
        raise ValueError('Four trajectories did not hash the same complete actual training input stream')
    common_fields=('micro_index','address','channel','standard_noise','pilot_polarity','gamma')
    for step in sorted(expected_steps):
        for micro in range(cfg.ACCUMULATION):
            items=[selected[cfg.key(a,m)][step]['input_fingerprints'][micro]
                for a in cfg.ARMS for m in cfg.TRAIN_METHODS]
            if any(any(item[name]!=items[0][name] for name in common_fields) for item in items[1:]):
                raise ValueError('Training snapshots are not paired across all four trajectories')
            for arm in cfg.ARMS:
                pair=[selected[cfg.key(arm,m)][step]['input_fingerprints'][micro] for m in cfg.TRAIN_METHODS]
                if pair[0]!=pair[1]:
                    raise ValueError('Receiver controls changed the frozen same-arm paid training inputs')
        ends=[selected[cfg.key(a,m)][step]['training_trace'] for a in cfg.ARMS for m in cfg.TRAIN_METHODS]
        if any(value!=ends[0] for value in ends[1:]):
            raise ValueError('Accepted training-boundary trace endpoints differ across trajectories')
    return dict(format=cfg.NAMESPACE+'_training_pairing_v1',status='passed',production_evidence=True,
        full_micro_trace_chain_verified=True,snapshot_pairing_verified=True,
        hashed_microbatches=cfg.UPDATES*cfg.ACCUMULATION,
        hashed_MU_frames=cfg.UPDATES*cfg.ACCUMULATION*cfg.MICRO_BATCH,
        logged_boundary_updates=len(expected_steps),logged_microbatches=len(expected_steps)*cfg.ACCUMULATION,
        gamma_counts=cfg.counters(cfg.UPDATES)['gamma_counts'],common_training_trace=traces[0],
        policy='last row per saved boundary; failed-attempt rows retained; digest is not tensor replay',
        trajectories=records)


def evaluate(output_dir, *, require_cuda=True, source_loader=None,terminal_loader=None,
        data_factory=None,forward=None):
    import torch
    cfg,runtime,results=_config(),_runtime(),_results()
    output=checked_output(output_dir)
    core=runtime.core_identity();runtime.guard_core(core)
    fixture=not require_cuda or any(x is not None for x in (source_loader,terminal_loader,data_factory,forward))
    device=runtime.configure_cuda() if require_cuda else torch.device('cpu')
    source_loader=source_loader or runtime.load_initial
    terminal_loader=terminal_loader or _trainer().load_finetuned
    data_factory=data_factory or runtime.make_data
    forward=forward or runtime.forward_codec
    calibration,calibration_record=runtime.load_calibration()
    locked_identity=_calibration_identity()
    locked_model=cfg.plain(calibration)
    # No final data object or channel is constructed until all four terminals pass.
    terminals={}
    for arm in cfg.ARMS:
        for method in cfg.TRAIN_METHODS:
            codec,stats,ce,record=terminal_loader(arm,method,device)
            _check_terminal_record(record,arm,method,fixture=fixture)
            _records_signature(codec,stats,ce)
            terminals[cfg.key(arm,method)]=cfg.plain(record)
            del codec,stats,ce
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    training_pairing=verify_training_pairing(terminals,fixture=fixture)
    output.mkdir(parents=True,exist_ok=False)
    manifest=_base_manifest(fixture,core)
    manifest.update(calibration_record=calibration_record,
        calibration_model_sha256=locked_identity['model_sha256'],
        calibration_manifest_sha256=locked_identity['manifest_sha256'],
        calibration_model_sha256_before=locked_identity['model_sha256'],
        terminal_records=terminals,training_pairing=training_pairing)
    started=time.perf_counter()
    rows,fingerprints,records,signatures,encoders=[],{}, {},{},{}
    protocol=None
    try:
        with torch.inference_mode():
            for phase in cfg.PHASES:
                records[phase],signatures[phase],encoders[phase]={},{},{}
                for arm in cfg.ARMS:
                    for method in cfg.TRAIN_METHODS:
                        loader=source_loader if phase=='source' else terminal_loader
                        codec,stats,ce,record=loader(arm,method,device)
                        if phase=='final':
                            _check_terminal_record(record,arm,method,fixture=fixture)
                            if cfg.plain(record)!=terminals[cfg.key(arm,method)]:
                                raise ValueError('Preflight terminal changed before final evaluation')
                        _check_files(record,fixture=fixture)
                        before=_records_signature(codec,stats,ce)
                        if before['CE']!=calibration_record['common_CE_signature']:
                            raise ValueError('Frozen CE differs from the pinned common calibrator source')
                        key=cfg.key(arm,method)
                        records[phase][key]=record
                        encoders[phase][key]=runtime.state_digest(codec.encoder)
                        data,metadata=data_factory(device),runtime.make_data(device)
                        if data.protocol!=metadata.protocol or protocol is not None and data.protocol!=protocol:
                            raise ValueError('Final data factory protocol identity changed')
                        protocol=data.protocol
                        for bi in range(cfg.FINAL_BATCHES):
                            absolute=cfg.ROLE_RANGES['final'][0]+bi
                            sample=data.sample('validation',absolute)
                            if sample.get('address')!=metadata.address('validation',absolute):
                                raise ValueError('Final data address changed to training/monitor/test')
                            truth_before=(_tensor_hash(sample['H_dl']),_tensor_hash(sample['H_ul']))
                            for gi,gamma in enumerate(cfg.GAMMAS):
                                if calibration!=locked_model:
                                    raise ValueError('Frozen calibration model changed in memory')
                                out=forward(codec,stats,ce,sample,gamma_db=gamma,
                                    noise_key=('final',bi,gi),method=method,calibration=calibration)
                                out.setdefault('layout',stats.layout)
                                _add_fingerprint(fingerprints,phase,arm,method,bi,sample,out,device)
                                output_hash=_tensor_hash(out['H_hat'])
                                recovered_before=_recovery_hashes(out)
                                vectors=score_out(out,sample['H_dl'])
                                _validate_vectors(vectors,cfg.MICRO_BATCH)
                                if (output_hash!=_tensor_hash(out['H_hat'])
                                        or recovered_before!=_recovery_hashes(out)
                                        or calibration!=locked_model):
                                    raise ValueError('Offline scoring mutated decoder/receiver/calibration inputs')
                                values={name:value.detach().cpu().tolist() for name,value in vectors.items()}
                                for within in range(cfg.MICRO_BATCH):
                                    rows.append(dict(role='final',phase=phase,arm=arm,method=method,
                                        gamma_db=float(gamma),batch_index=bi,within_batch=within,
                                        sample_index=bi*cfg.MICRO_BATCH+within,
                                        **{name:value[within] for name,value in values.items()}))
                            if truth_before!=(_tensor_hash(sample['H_dl']),_tensor_hash(sample['H_ul'])):
                                raise ValueError('Final offline scores mutated simulator truth')
                            runtime.atomic_json(output/'progress.json',dict(phase=phase,arm=arm,method=method,
                                completed_batches=bi+1,total_batches=cfg.FINAL_BATCHES,
                                updated=runtime.now(),neural_updates=0))
                            print(f'{phase}/{method}/{arm}: final batch {bi+1}/{cfg.FINAL_BATCHES}',flush=True)
                        after=_records_signature(codec,stats,ce)
                        if before!=after:raise ValueError('Final evaluation changed frozen model state')
                        signatures[phase][key]=dict(before=before,after=after)
                        _check_files(record,fixture=fixture)
                        del codec,stats,ce,data,sample,out
                        gc.collect()
                        if device.type=='cuda':torch.cuda.empty_cache()
        for arm in cfg.ARMS:
            if signatures['source'][cfg.key(arm,'original')]['before']['codec']!=signatures['source'][cfg.key(arm,'moment_receiver')]['before']['codec']:
                raise ValueError('Receiver methods did not share the same source initialization')
            hashes={encoders[p][cfg.key(arm,m)] for p in cfg.PHASES for m in cfg.TRAIN_METHODS}
            if len(hashes)!=1:raise ValueError('Decoder adaptation changed a source UE encoder')
            for name in ('CE','statistics'):
                if len({signatures[p][cfg.key(arm,m)]['before'][name]
                        for p in cfg.PHASES for m in cfg.TRAIN_METHODS})!=1:
                    raise ValueError('Decoder adaptation changed a frozen CE/prior state')
        entries=list(fingerprints.values())
        fingerprint_map(entries,device=device)
        _check_calibration_disjoint(entries)
        if _calibration_identity()!=locked_identity or calibration!=locked_model:
            raise ValueError('Pinned calibration changed during final evaluation')
        summary=results.summarize_rows(rows,samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,batch_size=cfg.MICRO_BATCH)
        comparisons=results.paired_comparisons(rows,samples=cfg.FINAL_BATCHES*cfg.MICRO_BATCH,batch_size=cfg.MICRO_BATCH)
        _write_gzip(output/'samples.csv.gz',rows)
        write_union_csv(output/'summary.csv',summary)
        write_union_csv(output/'comparisons.csv',comparisons)
        runtime.atomic_json(output/'input_fingerprints.json',entries,exclusive=True)
        runtime.atomic_json(output/'source_records.json',records,exclusive=True)
        runtime.atomic_json(output/'training_pairing.json',training_pairing,exclusive=True)
        runtime.guard_core(core)
        manifest.update(status='test_fixture' if fixture else 'passed',data_protocol=protocol,
            paired_inputs_verified=True,frozen_models_verified=True,source_initialization_matched=True,
            source_records=records,frozen_signatures=signatures,encoder_signatures=encoders,
            calibration_model_sha256_after=locked_identity['model_sha256'],calibration_frozen=True,
            elapsed_seconds=time.perf_counter()-started,
            files={name:runtime.sha256(output/name) for name in EVALUATION_FILES})
        runtime.atomic_json(output/'manifest.json',manifest,exclusive=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',
            elapsed_seconds=time.perf_counter()-started)
        runtime.atomic_json(output/'manifest.json',manifest)
        raise


def verify_manifest(directory, *, allow_fixture=False):
    cfg,runtime=_config(),_runtime()
    directory=checked_output(directory)
    manifest=runtime.read_json(directory/'manifest.json')
    fixture=manifest.get('status')=='test_fixture'
    if fixture and not allow_fixture:
        raise ValueError('CPU fixture cannot serve as production final evidence')
    expected=_base_manifest(fixture,runtime.core_identity())
    expected['status']='test_fixture' if fixture else 'passed'
    for name,value in expected.items():
        actual=manifest.get(name)
        if actual!=value or isinstance(value,(bool,int)) and type(actual) is not type(value):
            raise ValueError('Final evaluation identity/budget changed: '+name)
    runtime.guard_core(manifest['core'])
    if set(manifest.get('files',{}))!=set(EVALUATION_FILES):
        raise ValueError('Exact final scalar artifact inventory required')
    for name in EVALUATION_FILES:
        path=runtime.inside(directory/name)
        if path.is_symlink() or not path.is_file() or runtime.sha256(path)!=manifest['files'][name]:
            raise ValueError('Final evaluation artifact changed: '+name)
    identity=_calibration_identity()
    if (manifest.get('calibration_model_sha256')!=identity['model_sha256']
            or manifest.get('calibration_manifest_sha256')!=identity['manifest_sha256']
            or manifest.get('calibration_model_sha256_before')!=identity['model_sha256']
            or manifest.get('calibration_model_sha256_after')!=identity['model_sha256']
            or manifest.get('calibration_frozen') is not True):
        raise ValueError('Final pinned calibration lock changed')
    _,calibration_record=runtime.load_calibration()
    if manifest.get('calibration_record')!=calibration_record:
        raise ValueError('Calibration source evidence changed')
    records=runtime.read_json(directory/'source_records.json')
    expected_keys={cfg.key(a,m) for a in cfg.ARMS for m in cfg.TRAIN_METHODS}
    if records!=manifest.get('source_records') or set(records)!=set(cfg.PHASES):
        raise ValueError('Complete source/final record identity required')
    terminals=manifest.get('terminal_records',{})
    if set(terminals)!=expected_keys:
        raise ValueError('Four fixed terminal records required')
    paired=runtime.read_json(directory/'training_pairing.json')
    if (manifest.get('training_pairing')!=paired
            or paired!=verify_training_pairing(terminals,fixture=fixture)):
        raise ValueError('Final training pairing evidence changed')
    signatures=manifest.get('frozen_signatures',{})
    encoders=manifest.get('encoder_signatures',{})
    for phase in cfg.PHASES:
        if (set(records[phase])!=expected_keys or set(signatures.get(phase,{}))!=expected_keys
                or set(encoders.get(phase,{}))!=expected_keys
                or any(not _valid_hash(v) for v in encoders[phase].values())):
            raise ValueError('Complete per-model frozen evidence required')
        for arm in cfg.ARMS:
            for method in cfg.TRAIN_METHODS:
                key=cfg.key(arm,method)
                record=records[phase][key]
                _check_files(record,fixture=fixture)
                state=signatures[phase][key]
                if (set(state)!= {'before','after'} or state['before']!=state['after']
                        or set(state['before'])!={'codec','statistics','CE'}
                        or any(not _valid_hash(v) for v in state['before'].values())
                        or state['before']['CE']!=calibration_record['common_CE_signature']):
                    raise ValueError('Frozen codec/CE/prior signature changed')
                if phase=='final':
                    _check_terminal_record(record,arm,method,fixture=fixture)
                    if record!=terminals[key]:
                        raise ValueError('Fixed terminal preflight/source record changed')
                    if not fixture:
                        _verify_terminal_metadata(arm,method,record)
                        if record.get('source_identity')!=records['source'][key]:
                            raise ValueError('Adapted terminal no longer binds its source initialization')
                if not fixture:
                    for name,field in (('codec','state_sha256'),('CE','CE_state_sha256'),
                            ('statistics','statistics_state_sha256')):
                        if record.get(field)!=state['before'][name]:
                            raise ValueError('Frozen signatures do not bind actual loaded model records')
    for arm in cfg.ARMS:
        if len({encoders[p][cfg.key(arm,m)] for p in cfg.PHASES for m in cfg.TRAIN_METHODS})!=1:
            raise ValueError('UE encoder identity differs between paired adaptations')
        original=signatures['source'][cfg.key(arm,'original')]['before']['codec']
        moment=signatures['source'][cfg.key(arm,'moment_receiver')]['before']['codec']
        if original!=moment:
            raise ValueError('Source decoder initialization differs between receiver methods')
        for name in ('statistics','CE'):
            if len({signatures[p][cfg.key(arm,m)]['before'][name]
                    for p in cfg.PHASES for m in cfg.TRAIN_METHODS})!=1:
                raise ValueError('Adaptation changed its frozen CE/historical prior')
    for name in ('paired_inputs_verified','frozen_models_verified','source_initialization_matched'):
        if manifest.get(name) is not True:
            raise ValueError('Required final pairing/freezing evidence missing')
    device=manifest.get('data_protocol',{}).get('execution',{}).get('device')
    if manifest.get('data_protocol')!=runtime.make_data(device).protocol:
        raise ValueError('Final replay data protocol changed')
    entries=runtime.read_json(directory/'input_fingerprints.json')
    fingerprint_map(entries,device=device)
    _check_calibration_disjoint(entries)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    if not args.run:parser.print_help();return 0
    if args.output_dir is None:parser.error('--run requires --output-dir')
    evaluate(args.output_dir)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
