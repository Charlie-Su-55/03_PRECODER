#!/usr/bin/env python3
"""Paired, independently calibrated CE-only validation; no codec or rate run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from configs import twc_v1_k8_ce_refine_config as cfg
import twc_v1_k8_ce_refine as ce_math
import twc_v1_k8_ce_refine_results as statistics

VARIANTS = statistics.DEFAULT_VARIANTS
PUBLIC_FILES = ('README.md', 'samples.csv.gz', 'summary.csv', 'comparisons.csv',
    'input_fingerprints.json', 'provenance.json', 'calibration.json', 'curve_summary.csv')
MONITOR_FILES = ('summary.csv', 'input_fingerprints.json')
README_TEXT = '''# Independent pilot-CE continuation\n\nTwo pilot-only U-Net trajectories start from the same immutable CE5000 source and each receive60000 new CE updates. NMSE continuation and NMSE+direction+inter-user-Gram training use matched physical inputs and equal budgets. No UE, BS codec, FDMA or historical-power EMA is trained. Source files stay unchanged.\n\nReports contain complete-MU scalar observations and public fingerprints, never trained weights, channel tensors or credentials. Independent calibration is locked before fixed-terminal validation at six trained SNR nodes and five off-node interpolation points omitted by this six-node continuation. The original CE5000 source was trained over continuous Uniform[0,25] SNR; these midpoint checks are not globally unseen-SNR evaluation. Test remains sealed.\n\nThe shared recovery probe is an offline analytical conditional MSE under independent uniform-power symbols (power48/44), using filters designed from estimated UL and calibrated CE variance only. True UL is scoring-only. This is not a measured codec NMSE, downlink sum-rate or deployable oracle arm. Error improvement does not establish rate improvement.\n\nIntervals condition on fixed models and calibration. Gamma views are averaged within each MU before uncertainty over batches of4; calibration-refit and training-seed uncertainty are excluded. Negative results remain.\n'''


def _runtime():
    import ce_refine_runtime as runtime
    return runtime


def _digest(value):
    return cfg.digest(value)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _json(path, value, *, exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf-8') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        return
    fd,name=tempfile.mkstemp(prefix='report_',suffix='.json',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def _csv(path, rows, *, fields=None):
    path=Path(path);rows=list(rows)
    names=list(fields if fields is not None else rows[0].keys() if rows else ())
    if not names: raise ValueError('Declared CSV fields required even for an empty fixture curve')
    fd,name=tempfile.mkstemp(prefix='report_',suffix='.csv',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as binary:
            if path.suffix=='.gz':
                with gzip.GzipFile(filename='',mode='wb',fileobj=binary,mtime=0) as zipped:
                    with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as text:
                        writer=csv.DictWriter(text,fieldnames=names);writer.writeheader();writer.writerows(rows)
            else:
                with io.TextIOWrapper(binary,encoding='utf-8',newline='',write_through=True) as text:
                    writer=csv.DictWriter(text,fieldnames=names);writer.writeheader();writer.writerows(rows)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def _tensor_hash(value):
    return _runtime().original.state_digest({'tensor':value})


def _public_record(record):
    allowed=('checkpoint_sha256','source_checkpoint_sha256','source_state_sha256',
        'model_state_sha256','state_sha256','source_commit','source_core','source_updates','core',
        'step','CE_total_updates','counters','fingerprint','production_evidence')
    result={key:record[key] for key in allowed if key in record}
    source=record.get('source_identity',{})
    for key in ('source_state_sha256','source_commit'):
        if key not in result and key in source:result[key]=source[key]
    if 'checkpoint_sha256' in source:result['source_checkpoint_sha256']=source['checkpoint_sha256']
    if 'CE_total_updates' not in result and 'CE_total_updates' in record.get('counters',{}):
        result['CE_total_updates']=record['counters']['CE_total_updates']
    return result


@contextmanager
def frozen_models(models):
    """Freeze without leaking modes/flags; reject accidental parameter/buffer mutation."""
    import torch
    rt=_runtime()
    modes={name:{key:module.training for key,module in model.named_modules()} for name,model in models.items()}
    flags={name:[p.requires_grad for p in model.parameters()] for name,model in models.items()}
    before={name:rt.original.state_digest(model) for name,model in models.items()}
    try:
        for model in models.values():model.eval().requires_grad_(False)
        with torch.no_grad():yield before
    finally:
        after={name:rt.original.state_digest(model) for name,model in models.items()}
        for name,model in models.items():
            for key,module in model.named_modules():module.training=modes[name][key]
            for parameter,flag in zip(model.parameters(),flags[name]):parameter.requires_grad_(flag)
        if after!=before:raise ValueError('Frozen CE model or buffer changed during evaluation')


def _condition(rt, sample, role, noise_index, gamma, models, forward):
    """Evaluate one identical pilot observation across already-loaded CE models."""
    device=sample['H_ul'].device
    g,noise,polarity=rt.batch_random(role,noise_index,device,gamma=gamma)
    basic=dict(channel=sample['sample_fingerprint'],UL_channel=_tensor_hash(sample['H_ul']),address=sample['address'],
        standard_noise=_tensor_hash(noise),pilot_polarity=_tensor_hash(polarity),gamma=_tensor_hash(g))
    out, common = {}, None
    for variant,model in models.items():
        pred,initial,Y,pilots=forward(model,sample,g,noise,polarity)
        observed=dict(basic,observed_Y=_tensor_hash(Y),known_pilots=_tensor_hash(pilots),initial_LS=_tensor_hash(initial))
        if common is not None and observed!=common:
            raise ValueError('CE models did not receive identical physical pilot inputs')
        common=observed
        out[variant]=pred
    return out, common, g



def _verify_disjoint_inputs(inputs):
    by_role={}
    noises=set()
    conditions=set()
    for entry in inputs:
        role=entry['role'];by_role.setdefault(role,set()).add(entry['UL_channel'])
        key=(role,entry['batch_index'],float(entry['gamma_db']))
        if key in conditions:raise ValueError('Repeated CE physical condition')
        conditions.add(key)
        # Each role/batch/gamma has its own declared standard-noise address.
        if entry['standard_noise'] in noises:raise ValueError('CE noise realization reused across conditions')
        noises.add(entry['standard_noise'])
    roles=sorted(by_role)
    for i,first in enumerate(roles):
        for second in roles[i+1:]:
            if by_role[first]&by_role[second]:
                raise ValueError('CE calibration/final/midpoint actual UL channels overlap')
    return dict(cross_role_UL_collisions=0,unique_noise_conditions=len(noises))


def verify_monitor(output_dir, *, trajectory=None, step=None, production_required=True):
    path=Path(output_dir)
    if path.is_symlink():raise ValueError('Linked monitor directory refused')
    record=json.loads((path/'manifest.json').read_text())
    if (record.get('namespace')!=cfg.NAMESPACE or record.get('status')!='completed'
            or record.get('trajectory') not in cfg.TRAJECTORIES
            or type(record.get('step')) is not int or not 0<=record['step']<=cfg.UPDATES
            or record['step']%cfg.MONITOR_EVERY
            or record.get('fingerprint')!=cfg.fingerprint(record['trajectory'])
            or set(record.get('files',{}))!=set(MONITOR_FILES)):
        raise ValueError('Invalid CE monitor receipt')
    if trajectory is not None and record['trajectory']!=trajectory:raise ValueError('Monitor trajectory changed')
    if step is not None and record['step']!=step:raise ValueError('Monitor update changed')
    if production_required and record.get('production_evidence') is not True:
        raise ValueError('CPU monitor fixture cannot become production evidence')
    for name,digest in record['files'].items():
        f=path/name
        if f.is_symlink() or not f.is_file() or _sha(f)!=digest:raise ValueError('Monitor file/hash changed')
    if production_required:_runtime().guard_core(record['core'])
    return record


def monitor_model(model,data,trajectory,step,output_dir,*,production_evidence=True,forward=None,batches=None):
    """Same fixed monitor data at every checkpoint; never choose a best epoch."""
    rt=_runtime();cfg.key(trajectory)
    if type(step) is not int or not 0<=step<=cfg.UPDATES or step%cfg.MONITOR_EVERY:
        raise ValueError('Declared CE monitor boundary required')
    fixture=not production_evidence or forward is not None or batches is not None
    production_evidence=not fixture
    count=cfg.MONITOR_BATCHES if batches is None else batches
    if type(count) is not int or count<1:raise ValueError('Positive complete monitor batches required')
    path=rt.inside(output_dir) if production_evidence else Path(output_dir)
    if (path/'manifest.json').exists():
        record=verify_monitor(path,trajectory=trajectory,step=step,production_required=production_evidence)
        if record['model_state_sha256']!=rt.original.state_digest(model):raise ValueError('Monitor model state differs from checkpoint')
        return record
    path.mkdir(parents=True,exist_ok=True)
    core=rt.core_identity() if production_evidence else dict(schema='CPU_fixture',sha256={})
    before_rng=rt.original.capture_rng()
    rows,inputs=[],[];forward=rt.pilot_forward if forward is None else forward
    try:
        with frozen_models({trajectory:model}) as before:
            for bi in range(count):
                sample=data.sample('validation',bi)
                if sample['address']!=data.address('validation',bi):raise ValueError('Monitor data address changed')
                for gamma in cfg.GAMMAS:
                    gi=cfg.EVAL_GAMMAS.index(gamma)
                    estimates,physical,_=_condition(rt,sample,'monitor',bi*len(cfg.EVAL_GAMMAS)+gi,gamma,
                        {trajectory:model},forward)
                    scores=ce_math.vectors(estimates[trajectory],sample['H_ul'])
                    for j in range(cfg.MICRO_BATCH):
                        rows.append(dict(variant=trajectory,role='final',gamma_db=float(gamma),sample_index=bi*cfg.MICRO_BATCH+j,
                            batch_index=bi,within_batch=j,**{k:float(v[j].detach().cpu()) for k,v in scores.items()}))
                    inputs.append(dict(batch_index=bi,gamma_db=gamma,**physical))
            summaries=statistics.summarize_rows(rows,samples=count*cfg.MICRO_BATCH,
                batch_size=cfg.MICRO_BATCH,variants=(trajectory,))
            for row in summaries:row.update(role='monitor',trajectory=trajectory,step=step,CE_total_updates=cfg.SOURCE_UPDATES+step)
            _csv(path/'summary.csv',summaries)
            _json(path/'input_fingerprints.json',inputs)
    finally:
        rt.original.restore_rng(before_rng)
    if production_evidence:rt.guard_core(core)
    record=dict(namespace=cfg.NAMESPACE,status='completed',trajectory=trajectory,step=step,
        CE_total_updates=cfg.SOURCE_UPDATES+step,fingerprint=cfg.fingerprint(trajectory),core=core,
        model_state_sha256=before[trajectory],production_evidence=production_evidence,
        data_protocol=data.protocol,samples_per_node=count*cfg.MICRO_BATCH,selection=False,
        files={name:_sha(path/name) for name in MONITOR_FILES},updated=rt.original.now())
    _json(path/'manifest.json',record,exclusive=True)
    return record


def _interpolated_error(table,gamma,batch,device):
    import torch
    values=torch.as_tensor(table,dtype=torch.float32,device=device)
    if values.ndim!=3 or values.shape[0]!=6:raise ValueError('Six-node per-RE CE variance required')
    lower=min(int(float(gamma)//5),4);weight=(float(gamma)-5*lower)/5
    result=(1-weight)*values[lower]+weight*values[lower+1]
    return result[None].expand(batch,-1,-1)


def _curve_rows(*,production_required):
    rows=[]
    for trajectory in cfg.TRAJECTORIES:
        if production_required:
            import train_twc_v1_k8_ce_refine as trainer
            trainer.verify_terminal(trajectory)  # binds each curve model to its immutable checkpoint
        for step in range(0,cfg.UPDATES+1,cfg.MONITOR_EVERY):
            path=cfg.OUTPUT_ROOT/'training'/trajectory/'monitor'/f'step_{step:05d}'
            if not (path/'manifest.json').exists():
                if production_required:raise FileNotFoundError('Complete learning curve missing: '+str(path))
                continue
            verify_monitor(path,trajectory=trajectory,step=step,production_required=production_required)
            with (path/'summary.csv').open(newline='') as stream:rows.extend(csv.DictReader(stream))
    return rows


def run_evaluation(output_dir,*,require_cuda=True,models=None,data=None,forward=None,
        final_batches=None,calibration_batches=None,midpoint_batches=None,probe_layout=None):
    """Calibrate then validate all fixed models; injections create labelled CPU fixtures."""
    import torch
    rt=_runtime()
    fixture=(not require_cuda or any(x is not None for x in
        (models,data,forward,final_batches,calibration_batches,midpoint_batches,probe_layout)))
    production=not fixture
    device=rt.configure_cuda() if require_cuda else torch.device('cpu')
    core=rt.core_identity() if production else dict(schema='CPU_fixture',sha256={})
    if models is None:
        import train_twc_v1_k8_ce_refine as trainer
        source,source_record=rt.load_source(device)
        nmse,nmse_record=trainer.load_terminal('nmse',device)
        geometry,geometry_record=trainer.load_terminal('geometry',device)
        models={'source_ce5000':(source,source_record),'nmse65000':(nmse,nmse_record),
            'geometry65000':(geometry,geometry_record)}
    if set(models)!=set(VARIANTS):raise ValueError('All three fixed CE models must load before data generation')
    networks={k:v[0] for k,v in models.items()};records={k:v[1] for k,v in models.items()}
    # Constructing a replay stream happens only after all strict loaders return.
    data=rt.make_data(device) if data is None else data
    forward=rt.pilot_forward if forward is None else forward
    nfinal=cfg.FINAL_BATCHES if final_batches is None else final_batches
    ncal=cfg.CALIBRATION_BATCHES if calibration_batches is None else calibration_batches
    nmid=cfg.MIDPOINT_BATCHES if midpoint_batches is None else midpoint_batches
    if any(type(n) is not int or n<1 for n in (nfinal,ncal,nmid)):
        raise ValueError('Positive fixed complete batch counts required')
    layout=rt.physics().make_layout('hybrid_g4') if probe_layout is None else probe_layout
    path=rt.inside(output_dir) if production else Path(output_dir)
    path.mkdir(parents=True,exist_ok=False)
    inputs=[];raw=[]
    before_rng=rt.original.capture_rng()
    with frozen_models(networks) as frozen_before:
        try:
            # An independent calibration role, same actual pilot observations across models.
            totals={key:None for key in VARIANTS}
            for bi in range(ncal):
                sample=data.sample('ce_calibration',bi)
                if sample['address']!=data.address('ce_calibration',bi):raise ValueError('Calibration address changed')
                for ni,gamma in enumerate(cfg.GAMMAS):
                    gi=cfg.EVAL_GAMMAS.index(gamma)
                    est,physical,_=_condition(rt,sample,'calibration',bi*len(cfg.EVAL_GAMMAS)+gi,gamma,networks,forward)
                    inputs.append(dict(role='calibration',batch_index=bi,gamma_db=gamma,**physical))
                    for key in VARIANTS:
                        per_re=(est[key]-sample['H_ul']).abs().square().mean((0,2))
                        if totals[key] is None:totals[key]=[None]*len(cfg.GAMMAS)
                        totals[key][ni]=per_re if totals[key][ni] is None else totals[key][ni]+per_re
            errors={key:[(row/ncal).cpu().tolist() for row in totals[key]] for key in VARIANTS}
            calibration=dict(namespace=cfg.NAMESPACE,production_evidence=production,nodes=list(cfg.GAMMAS),
                per_RE_error=errors,samples_per_node=ncal*cfg.MICRO_BATCH,model_states=frozen_before,
                definition='absolute complex error power per UE and RE, averaged independent calibration MU/antenna',
                interpolation='nonnegative linear in public gamma;no extrapolation',data_protocol=data.protocol,
                input_fingerprint_sha256=_digest(inputs),locked_before_final=True)
            calibration['record_id']=_digest(calibration)
            _json(path/'calibration.json',calibration,exclusive=True)
            calibration_sha=_sha(path/'calibration.json')
            for role,n,first,gammas in (
                ('final',nfinal,cfg.FINAL_START,cfg.GAMMAS),
                ('midpoint',nmid,cfg.MIDPOINT_START,cfg.MIDPOINT_GAMMAS)):
                for bi in range(n):
                    sample=data.sample('validation',first+bi)
                    if sample['address']!=data.address('validation',first+bi):raise ValueError('Fixed validation address changed')
                    for gamma in gammas:
                        gi=cfg.EVAL_GAMMAS.index(gamma)
                        est,physical,g=_condition(rt,sample,role,bi*len(cfg.EVAL_GAMMAS)+gi,gamma,networks,forward)
                        inputs.append(dict(role=role,batch_index=bi,gamma_db=gamma,**physical))
                        for variant in VARIANTS:
                            scores=ce_math.vectors(est[variant],sample['H_ul'])
                            variance=_interpolated_error(errors[variant],gamma,cfg.MICRO_BATCH,est[variant].device)
                            scores.update(ce_math.shared_uniform_prior_score(est[variant],sample['H_ul'],variance,g,layout))
                            for j in range(cfg.MICRO_BATCH):
                                raw.append(dict(variant=variant,role=role,gamma_db=float(gamma),
                                    sample_index=bi*cfg.MICRO_BATCH+j,batch_index=bi,within_batch=j,
                                    **{k:float(v[j].detach().cpu()) for k,v in scores.items()}))
                print(f'CE validation {role}: {n*cfg.MICRO_BATCH} MU per gamma, all models',flush=True)
            disjoint=_verify_disjoint_inputs(inputs)
            if _sha(path/'calibration.json')!=calibration_sha:raise ValueError('Locked CE calibration changed')
            frozen_after={k:rt.original.state_digest(v) for k,v in networks.items()}
            if frozen_after!=frozen_before:raise ValueError('Frozen CE state changed')
            if production:
                rt.guard_core(core);rt.verify_source(records['source_ce5000'])
        finally:
            rt.original.restore_rng(before_rng)
    # No neural model calls or calibration fitting after fixed data aggregation.
    summaries=[];comparisons=[]
    for role,n in (('final',nfinal),('midpoint',nmid)):
        rows=[row for row in raw if row['role']==role]
        summaries.extend(statistics.summarize_rows(rows,samples=n*cfg.MICRO_BATCH,role=role))
        comparisons.extend(statistics.paired_comparisons(rows,samples=n*cfg.MICRO_BATCH,role=role))
    curves=_curve_rows(production_required=production)
    _csv(path/'samples.csv.gz',raw);_csv(path/'summary.csv',summaries);_csv(path/'comparisons.csv',comparisons)
    _csv(path/'curve_summary.csv',curves,fields=list(curves[0]) if curves else
        ('variant','role','scope','gamma_db','metric','mean','trajectory','step','CE_total_updates'))
    _json(path/'input_fingerprints.json',dict(namespace=cfg.NAMESPACE,production_evidence=production,
        all_models_share_each_input=True,records=inputs))
    provenance=dict(namespace=cfg.NAMESPACE,production_evidence=production,
        fixture=fixture,core=core,fingerprint=cfg.fingerprint(),data_protocol=data.protocol,
        source_models={k:_public_record(v) for k,v in records.items()},
        UE_updates=0,BS_updates=0,FDMA_updates=0,EMA_updates=0,CE_training_updates=2*cfg.UPDATES,
        fixed_terminal_new_updates=cfg.UPDATES,CE_total_updates_per_new_model=cfg.SOURCE_UPDATES+cfg.UPDATES,
        all_models_verified_before_data=True,paired_inputs_verified=True,
        freeze_before=frozen_before,freeze_after=frozen_after,
        calibration_sha256=calibration_sha,calibration_record_id=calibration['record_id'],
        calibration_locked_before_final=True,samples_per_gamma=dict(final=nfinal*cfg.MICRO_BATCH,midpoint=nmid*cfg.MICRO_BATCH),
        input_count=len(inputs),raw_rows=len(raw),test_used=False,disjoint_inputs=disjoint,
        performance_claim='CE metrics and uniform-prior analytical proxy only; no codec or downlink rate evaluation',
        uncertainty='fixed-model and fixed-calibration paired validation sampling; one source training seed',
        midpoint_definition='off-node interpolation not sampled by six-node continuation; source CE5000 had continuous Uniform[0,25] training',
        generated=rt.original.now())
    _json(path/'provenance.json',provenance)
    (path/'README.md').write_text(README_TEXT,encoding='utf-8')
    manifest=dict(namespace=cfg.NAMESPACE,status='completed',production_evidence=production,
        core=core,fingerprint=cfg.fingerprint(),recipe_fingerprint=cfg.fingerprint(),raw_rows=len(raw),
        files={name:_sha(path/name) for name in PUBLIC_FILES},updated=rt.original.now())
    _json(path/'manifest.json',manifest,exclusive=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    run_evaluation(args.output_dir)
    return 0


if __name__=='__main__':raise SystemExit(main())
