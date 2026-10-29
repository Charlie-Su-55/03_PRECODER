#!/usr/bin/env python3
"""One registered 266d9e5 JSON-provenance repair; not a generic hash bypass."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from as_receiver_batch_support import safe_inside,sha256,save_json
from configs.grouped_hybrid_anchor_train_config import OUTPUT_ROOT,ARMS,SEEDS,anchor_config
from train_grouped_codec import _json

ROOT=Path(__file__).resolve().parent
BASE_COMMIT='266d9e52428d136f9de5d44a95da4afeea812fb6'
BATCH_NAME='batch_cf66957ea2914720bbdd9ed15dbf3884'
BATCH=OUTPUT_ROOT/'batches'/BATCH_NAME
REGISTRATION=BATCH/'resume_compat_266d9e5'
RECORD=REGISTRATION/'record.json'
SNAPSHOT=REGISTRATION/'batch_summary.before.json'
SEAL=REGISTRATION/'record.sha256'
TOOL='grouped_hybrid_resume_compat.py'
CHANGED_PATHS=frozenset(('train_grouped_hybrid_anchor.py','run_grouped_hybrid_anchor_batch.py',
    TOOL,'tests/test_grouped_hybrid_resume_compat.py','docs/GROUPED_HYBRID_ANCHOR_BATCH.md'))
# Exact reviewed executable replacements, Git blob bytes (LF). Not a filename allowlist.
APPROVED_REPLACEMENTS={
    'train_grouped_hybrid_anchor.py':(
        'b060e62997c0afe8c6f87177f974a65c53a2c6e258dc9a0ca9d42ae93b600a5c',
        '825f2f93065ec0c1a9fdbf21e49f1f5586e97fa539f3b54ee3c76f527b45728d'),
    'run_grouped_hybrid_anchor_batch.py':(
        '32af90ab7e55ffedffe7233cca81425bb72fd9f94773f2eea6aed5baf31cd6b6',
        '5ec7c66c140071b1a4d59a337b55554d4297b9f447ad3c63736276d6888c99a3'),
}
COMPLETED=('environment','sources','cuda_preflight')
FAILED_TRAIN='train_fdma_control_20261030'


def canonical(value):return json.loads(_json(value))


def differences(old,new,path='$'):
    """Report semantic leaf changes without treating tuple/list serialization as drift."""
    old,new=canonical(old),canonical(new)
    if _json(old)==_json(new):return []
    if isinstance(old,dict) and isinstance(new,dict):
        result=[]
        for key in sorted(old.keys()|new.keys()):
            if key not in old or key not in new:
                result.append(dict(field=path+'.'+key,old=old.get(key,'<missing>'),new=new.get(key,'<missing>')))
            else:result.extend(differences(old[key],new[key],path+'.'+key))
        return result
    if isinstance(old,list) and isinstance(new,list) and len(old)==len(new):
        return [change for i,(a,b) in enumerate(zip(old,new)) for change in differences(a,b,f'{path}[{i}]')]
    return [dict(field=path,old=old,new=new)]


def _equal(old,new,label):
    changes=differences(old,new)
    if changes:raise ValueError(label+': '+json.dumps(changes,sort_keys=True))


def _git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def _blob(commit,name):return _git('show',f'{commit}:{name}')
def _digest_bytes(value):return hashlib.sha256(value).hexdigest()
def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_release_description(repair_commit,head,parents,changed,old_digests,new_digests):
    """Pure policy check, independently testable without server evidence."""
    if not isinstance(repair_commit,str) or not re.fullmatch('[0-9a-f]{40}',repair_commit):raise ValueError('Exact repair commit SHA required')
    if repair_commit==BASE_COMMIT or head!=repair_commit or parents!=[BASE_COMMIT]:
        raise ValueError('Repair must be the explicitly pinned direct child of 266d9e5 and current HEAD')
    if set(changed)!=CHANGED_PATHS:raise ValueError('Unexpected commit file changes: '+str(sorted(set(changed)^CHANGED_PATHS)))
    for name,(old,new) in APPROVED_REPLACEMENTS.items():
        if old_digests.get(name)!=old or new_digests.get(name)!=new:
            raise ValueError('Unreviewed executable bytes: '+name)


def release_identity(repair_commit):
    """The user-supplied release SHA pins this tool as well as every other file."""
    from train_grouped_hybrid_anchor import SOURCE_FILES
    from run_grouped_hybrid_anchor_batch import source_record
    if _git('branch','--show-current').decode().strip()!='main':raise ValueError('Expected main')
    if _git('remote','get-url','origin').decode().strip()!='https://github.com/Charlie-Su-55/03_PRECODER.git':raise ValueError('Unexpected origin')
    head=_git('rev-parse','HEAD').decode().strip()
    parents=_git('show','-s','--format=%P',repair_commit).decode().strip().split()
    changed=_git('diff-tree','--no-commit-id','--name-only','-r',repair_commit).decode().splitlines()
    old={name:_digest_bytes(_blob(BASE_COMMIT,name)) for name in APPROVED_REPLACEMENTS}
    new={name:_digest_bytes(_blob(repair_commit,name)) for name in APPROVED_REPLACEMENTS}
    validate_release_description(repair_commit,head,parents,changed,old,new)
    batch_now=source_record()['sha256']
    names=set(batch_now)|set(SOURCE_FILES)|CHANGED_PATHS
    tree={name:_digest_bytes(_blob(repair_commit,name)) for name in sorted(names)}
    for name,digest in tree.items():
        if sha256(safe_inside(ROOT/name,ROOT))!=digest:raise ValueError('Working file differs from pinned repair commit: '+name)
    old_batch={name:_digest_bytes(_blob(BASE_COMMIT,name)) for name in batch_now if name!=TOOL}
    new_batch={name:tree[name] for name in batch_now}
    old_train={name:_digest_bytes(_blob(BASE_COMMIT,name)) for name in SOURCE_FILES if name!=TOOL}
    new_train={name:tree[name] for name in SOURCE_FILES}
    for name,digest in old_batch.items():
        if name not in APPROVED_REPLACEMENTS and new_batch[name]!=digest:
            raise ValueError('Protected computational source changed: '+name)
    return dict(old_commit=BASE_COMMIT,repair_commit=repair_commit,reviewed_changes={name:dict(old=old[name],new=new[name]) for name in old},
        old_controller_sha256=old_batch,new_controller_sha256=new_batch,
        old_trainer_sha256=old_train,new_trainer_sha256=new_train,
        release_files_sha256=tree,compatibility_tool_sha256=tree[TOOL])


def _fixed_folder(folder):
    actual=safe_inside(folder,OUTPUT_ROOT/'batches')
    if actual!=BATCH.absolute():raise ValueError('Compatibility is restricted to '+BATCH_NAME)
    pointer=_read(OUTPUT_ROOT/'last_batch.json')
    if safe_inside(pointer['directory'],OUTPUT_ROOT/'batches')!=actual:raise ValueError('Batch pointer differs from exact failed batch')
    return actual


def inspect_completed_stages(summary):
    """No stage is rerun; every recorded file must still match its old digest."""
    immutable={}
    for key in COMPLETED:
        stage=summary.get('stages',{}).get(key,{})
        if stage.get('status')!='completed' or not stage.get('evidence_sha256'):
            raise ValueError('Missing completed evidence for '+key)
        report=stage.get('result',{}).get('report')
        if not isinstance(report,dict) or report.get('passed') is not True:raise ValueError('Stage did not pass: '+key)
        if key=='cuda_preflight' and report.get('commit')!=BASE_COMMIT:raise ValueError('Wrong CUDA acceptance commit')
        if key=='environment' and report.get('source',{}).get('commit')!=BASE_COMMIT:raise ValueError('Wrong environment commit')
        report_matches=0
        for path,digest in stage['evidence_sha256'].items():
            path=safe_inside(path,ROOT)
            if sha256(path)!=digest:raise ValueError('Completed evidence changed: '+str(path))
            immutable[str(path)]=digest
            if path.suffix=='.json' and canonical(_read(path))==canonical(report):report_matches+=1
        if not report_matches:raise ValueError('No intact report file matches embedded stage '+key)
    return immutable


def verify_paused_checkpoint():
    """Server-only read/strict load. Never construct an optimizer or save weights."""
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA server required for saved CUDA RNG validation; no laptop fallback')
    from train_grouped_hybrid_anchor import (check_run_state,load_source,validate_resume_payload,
        validate_best_metadata,build_model,validate_runtime_resume,validate_initialization_provenance,fingerprint)
    from evaluate_proposal_as_direct import check_tensor_state
    cfg=anchor_config('fdma_control',20261030);check_run_state(cfg,resume=True)
    status=_read(cfg.STATUS_PATH)
    if status.get('state')!='paused' or status.get('step')!=500:raise ValueError('Expected exactly paused fdma_control/20261030 step500')
    initial=_read(Path(cfg.LOG_DIR)/'initialization.json')
    if initial.get('git_commit')!=BASE_COMMIT:raise ValueError('Wrong initialization commit')
    for seed in SEEDS:
        for arm in ARMS:
            other=anchor_config(arm,seed)
            if (arm,seed)==('fdma_control',20261030):continue
            if Path(other.SAVE_DIR).exists() or Path(other.LOG_DIR).exists():raise ValueError('Another arm has already started: '+other.EXP_NAME)
    evidence_paths=[Path(cfg.SAVE_PATH),Path(cfg.BEST_PATH),Path(cfg.CONFIG_PATH),Path(cfg.STATUS_PATH),Path(cfg.METRICS_PATH)]
    before={str(p):sha256(p) for p in evidence_paths}
    latest=torch.load(cfg.SAVE_PATH,map_location='cpu',weights_only=False)
    best=torch.load(cfg.BEST_PATH,map_location='cpu',weights_only=False)
    if latest.get('step')!=500 or best.get('step')!=500:raise ValueError('Expected both best/latest step500')
    validate_resume_payload(latest,cfg);validate_best_metadata(best,latest,cfg)
    check_tensor_state(torch,best['state'],latest['state'],equal=True)
    model,_=build_model(torch,cfg.arm,cfg.SEED,best,torch.device('cpu'))
    validate_runtime_resume(torch,model,latest,cfg)
    validate_runtime_resume(torch,model,best,cfg)
    source,source_record=load_source(torch,20261030);del source
    # Exercise the actual trainer boundary, retaining the old code map for this
    # metadata check. Code-map authorization is separately proven by release_identity.
    validate_initialization_provenance(cfg,initial,dict(source=source_record,source_sha256=initial['source_sha256']))
    _equal(_read(cfg.CONFIG_PATH),dict(cfg.to_dict(),config_fingerprint=fingerprint(cfg)),'Saved config/fingerprint')
    points=[]
    for line in Path(cfg.METRICS_PATH).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        item=json.loads(line)
        if item.get('step',501)>500:raise ValueError('Metrics extend beyond update500')
        if item.get('event')=='validation':
            if item.get('selection_metric_name')!='mean_all':raise ValueError('Different selection criterion')
            points.append(item)
    if [p['step'] for p in points]!=[0,500] or max(points,key=lambda p:p['selection_metric'])['step']!=500:
        raise ValueError('Validation log does not select step500 from exactly step0/500')
    if abs(points[-1]['selection_metric']-best['best_metric'])>1e-10:raise ValueError('Logged/loaded best metric mismatch')
    if status.get('best_metric_name')!='mean_all' or status.get('best_metric')!=best['best_metric']:
        raise ValueError('Paused status/best selection mismatch')
    for path,digest in before.items():
        if sha256(path)!=digest:raise ValueError('Active evidence changed during read-only validation: '+path)
    return dict(arm=cfg.arm,seed=cfg.SEED,best_step=500,latest_step=500,next_step=501,total_steps=20000,
        prior_count=int(latest['state']['power_prior.update_count'].item()),strict_state=True,
        optimizer_scheduler_rng_verified=True,optimizer_constructed=False,weights_written=False,
        source_record=source_record,active_files_before=before,other_five_unstarted=True)


def _environment_matches(summary):
    # Environment inspection only, not the completed stage or any model preflight.
    from run_grouped_hybrid_anchor_batch import environment
    actual=environment();expected=summary['stages']['environment']['result']['report']
    for key in ('torch','sionna','matplotlib','channel_backend','cuda'):_equal(expected[key],actual[key],'Environment '+key)


def receipt(record):
    return dict(record_path=str(RECORD),record_sha256=sha256(RECORD),snapshot_path=str(SNAPSHOT),
        snapshot_sha256=record['snapshot_sha256'],version_chain=[BASE_COMMIT,record['release']['repair_commit']],
        completed_cuda_acceptance_commit=BASE_COMMIT,compatibility='exact JSON-provenance repair only')


def load_registration(path):
    path=safe_inside(path,BATCH)
    if path!=RECORD.absolute():raise ValueError('Only the exact registered compatibility path is accepted')
    _fixed_folder(BATCH)
    safe_inside(SEAL,BATCH);safe_inside(SNAPSHOT,BATCH)
    sealed=SEAL.read_text(encoding='ascii').strip()
    if not re.fullmatch('[0-9a-f]{64}',sealed) or sha256(path)!=sealed:raise ValueError('Compatibility record was changed or is incomplete')
    record=_read(path)
    if record.get('schema')!='grouped-hybrid-266d9e5-resume-repair-v1' or record.get('batch_name')!=BATCH_NAME:
        raise ValueError('Wrong compatibility schema/batch')
    release=release_identity(record['release']['repair_commit'])
    _equal(record['release'],release,'Registered exact source transition')
    if sha256(SNAPSHOT)!=record['snapshot_sha256']:raise ValueError('Pre-repair summary snapshot changed')
    original=_read(SNAPSHOT)
    _equal(original['source'],dict(commit=BASE_COMMIT,sha256=release['old_controller_sha256']),'Original batch provenance')
    _equal(record['completed_stages'],{key:original['stages'][key] for key in COMPLETED},'Original completed stages')
    required_immutable=inspect_completed_stages(original)
    initial_path=safe_inside(Path(anchor_config('fdma_control',20261030).LOG_DIR)/'initialization.json',ROOT)
    required_immutable[str(initial_path)]=sha256(initial_path)
    _equal(_read(initial_path).get('source_sha256'),release['old_trainer_sha256'],'Preserved initialization source hashes')
    for filename,digest in required_immutable.items():
        if record['immutable_files'].get(filename)!=digest:raise ValueError('Required immutable evidence missing: '+filename)
    for path,digest in record['immutable_files'].items():
        if sha256(safe_inside(path,ROOT))!=digest:raise ValueError('Immutable pre-repair evidence changed: '+path)
    expected=dict(arm='fdma_control',seed=20261030,best_step=500,latest_step=500,next_step=501,total_steps=20000,
        prior_count=500,strict_state=True,optimizer_scheduler_rng_verified=True,optimizer_constructed=False,
        weights_written=False,other_five_unstarted=True)
    for key,value in expected.items():_equal(record.get('verification',{}).get(key),value,'Registered checkpoint verification '+key)
    return record


def validate_controller_transition(folder,summary,current,path):
    _fixed_folder(folder);record=load_registration(path)
    release=record['release']
    _equal(summary['source'],dict(commit=BASE_COMMIT,sha256=release['old_controller_sha256']),'Original controller provenance must remain preserved')
    _equal(current,dict(commit=release['repair_commit'],sha256=release['new_controller_sha256']),'Current controller provenance')
    for key in COMPLETED:_equal(summary['stages'][key],record['completed_stages'][key],'Completed stage must not be replaced: '+key)
    result=receipt(record)
    if summary.get('resume_compatibility') is not None:_equal(summary['resume_compatibility'],result,'Compatibility version chain/record digest')
    else:
        if sha256(BATCH/'batch_summary.json')!=record['snapshot_sha256']:raise ValueError('Batch changed between registration and first resume')
        for filename,digest in record['verification']['active_files_before'].items():
            if sha256(filename)!=digest:raise ValueError('Checkpoint/status changed before first compatible resume: '+filename)
    _environment_matches(summary)
    return result


def validate_trainer_transition(cfg,initial,current,path):
    record=load_registration(path);release=record['release']
    if cfg.arm not in ARMS or cfg.SEED not in SEEDS:raise ValueError('Unregistered training identity')
    _equal(current['source_sha256'],release['new_trainer_sha256'],'Current trainer source-code SHA256')
    if current.get('git_commit')!=release['repair_commit']:raise ValueError('Trainer current commit differs')
    if initial.get('source_sha256')==release['old_trainer_sha256']:
        if (cfg.arm,cfg.SEED)!=('fdma_control',20261030) or initial.get('git_commit')!=BASE_COMMIT:
            raise ValueError('Only the existing500-step arm may use old initialization provenance')
        initial_path=str(Path(cfg.LOG_DIR)/'initialization.json')
        if initial_path not in record['immutable_files'] or sha256(initial_path)!=record['immutable_files'][initial_path]:
            raise ValueError('Original trainer initialization changed')
        _equal(initial,_read(initial_path),'Loaded old initialization record')
    else:
        _equal(initial.get('source_sha256'),release['new_trainer_sha256'],'Unregistered trainer source-code change')
        if initial.get('git_commit')!=release['repair_commit']:raise ValueError('New arm initialization commit differs')
    live=_read(BATCH/'batch_summary.json')
    _equal(live.get('resume_compatibility'),receipt(record),'Trainer must share controller compatibility receipt')
    return receipt(record)


def prepare(repair_commit):
    """Exclusive registration; no checkpoint write, optimizer creation or training."""
    if os.name!='posix':raise RuntimeError('Compatibility preparation must run on the evidence-owning CUDA server')
    release=release_identity(repair_commit);_fixed_folder(BATCH)
    from train_proposal_as_receiver import run_lock
    owner=SimpleNamespace(EXP_NAME='grouped_hybrid_anchor_batch_v1',arm='controller',SEED=0)
    with ExitStack() as stack:
        # The same locks as controller/trainer prove no cooperating process is alive.
        stack.enter_context(run_lock(owner,resume=True,allowed_root=OUTPUT_ROOT))
        for seed in SEEDS:
            for arm in ARMS:stack.enter_context(run_lock(anchor_config(arm,seed),resume=True,allowed_root=OUTPUT_ROOT))
        if RECORD.exists():
            existing=load_registration(RECORD)
            if existing['release']['repair_commit']!=repair_commit:raise ValueError('Already registered for a different repair')
            validate_controller_transition(BATCH,_read(BATCH/'batch_summary.json'),
                dict(commit=repair_commit,sha256=release['new_controller_sha256']),RECORD)
            print(json.dumps(dict(status='already_registered',**receipt(existing)),indent=2));return receipt(existing)
        if REGISTRATION.exists():raise FileExistsError('Incomplete registration preserved; refuse overwrite')
        before_bytes=(BATCH/'batch_summary.json').read_bytes();summary=json.loads(before_bytes)
        _equal(summary.get('source'),dict(commit=BASE_COMMIT,sha256=release['old_controller_sha256']),'Failed batch source identity')
        if summary.get('status')!='failed' or 'Resume code/source checkpoint provenance changed' not in summary.get('error','')+summary.get('traceback',''):
            # The controller often stores only the child's exit/log in its error.
            stage_error=summary.get('stages',{}).get(FAILED_TRAIN,{})
            if summary.get('status')!='failed' or stage_error.get('status')!='failed':raise ValueError('Expected the specific failed continuation stage')
        if set(summary['stages'])!=set(COMPLETED)|{FAILED_TRAIN} or summary['stages'][FAILED_TRAIN]['status']!='failed':
            raise ValueError('Unexpected completed/started training stages')
        immutable=inspect_completed_stages(summary)
        initial_path=Path(anchor_config('fdma_control',20261030).LOG_DIR)/'initialization.json'
        initial=_read(initial_path)
        _equal(initial.get('source_sha256'),release['old_trainer_sha256'],'Original trainer source identity')
        immutable[str(initial_path)]=sha256(initial_path)
        package=safe_inside(summary['result_package'],BATCH)
        if sha256(package)!=summary['result_package_sha256']:raise ValueError('Original result package changed')
        immutable[str(package)]=sha256(package)
        # Preserve all existing small batch/log evidence, including failed continuation logs.
        for parent in (BATCH,initial_path.parent):
            for path in parent.iterdir():
                if path.is_file() and path.suffix in ('.json','.log') and path.name not in ('batch_summary.json','status.json','config.json'):
                    path=safe_inside(path,ROOT);immutable[str(path)]=sha256(path)
        _environment_matches(summary)
        verified=verify_paused_checkpoint()
        _equal(initial.get('source'),verified['source_record'],'Original/current source metadata')
        for filename,digest in {**immutable,**verified['active_files_before']}.items():
            if sha256(filename)!=digest:raise ValueError('Evidence changed while preparing: '+filename)
        if (BATCH/'batch_summary.json').read_bytes()!=before_bytes:raise ValueError('Batch summary changed during preparation')
        record=dict(schema='grouped-hybrid-266d9e5-resume-repair-v1',batch_name=BATCH_NAME,
            created_utc=datetime.now(timezone.utc).isoformat(),release=release,
            snapshot_sha256=_digest_bytes(before_bytes),immutable_files=immutable,
            completed_stages={key:summary['stages'][key] for key in COMPLETED},verification=verified,
            note='Read-only real500-step verification; original evidence untouched; future active checkpoints remain subject to full strict resume checks')
        safe_inside(REGISTRATION,BATCH).mkdir(exist_ok=False)
        with SNAPSHOT.open('xb') as stream:stream.write(before_bytes)
        save_json(RECORD,record)
        with SEAL.open('x',encoding='ascii') as stream:stream.write(sha256(RECORD)+'\n')
        load_registration(RECORD)
        for filename,digest in verified['active_files_before'].items():
            if sha256(filename)!=digest:raise ValueError('Checkpoint/status changed during registration')
        result=receipt(record);print(json.dumps(dict(status='registered',verification=verified,**result),indent=2));return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--prepare',action='store_true')
    parser.add_argument('--repair-commit',help='Exact approved repair commit from the delivery command, not an arbitrary version')
    args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(base_commit=BASE_COMMIT,batch=str(BATCH),record=str(RECORD),
            expected_arm='fdma_control',expected_seed=20261030,best_step=500,latest_step=500,next_step=501,
            other_five='must not have started',checkpoint_reads=False,writes=False),indent=2));return 0
    if args.prepare:
        if args.repair_commit is None:parser.error('--prepare requires the exact --repair-commit')
        prepare(args.repair_commit);return 0
    parser.print_help();return 0


if __name__=='__main__':raise SystemExit(main())
