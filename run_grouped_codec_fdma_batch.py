#!/usr/bin/env python3
"""Isolated F x two seeds, archived-test completion, and independent replication."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
import uuid
import zipfile

from as_receiver_batch_support import safe_inside, save_json, sha256
from run_as_receiver_batch import _child, _run_stage, now, read_json
from train_proposal_as_receiver import run_lock
from configs.grouped_codec_fdma_train_config import ARMS, SEEDS, OUTPUT_ROOT, fdma_config

ROOT = Path(__file__).resolve().parent
BATCH_ROOT = OUTPUT_ROOT / 'batches'
EVAL_ROOT = ROOT / 'runs_as_screen/evaluation/k4_grouped_codec_fdma_v1'
SMALL_SUFFIXES = frozenset(('.csv', '.json', '.md', '.pdf', '.png'))
SMALL_FILE_LIMIT = 32 * 1024**2
SMALL_TOTAL_LIMIT = 128 * 1024**2


def source_record():
    from train_grouped_codec_fdma import SOURCE_FILES as training
    from evaluate_grouped_codec_fdma import SOURCE_FILES as evaluation
    files = tuple(dict.fromkeys(training + evaluation + (
        'run_grouped_codec_fdma_batch.py', 'preflight_grouped_codec_fdma.py',
        'grouped_codec_fdma_results.py', 'run_grouped_codec_batch.py',
        'run_as_receiver_batch.py', 'as_receiver_batch_support.py')))
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                sha256={name: sha256(ROOT / name) for name in files})


def plan():
    from evaluate_grouped_codec_fdma import DATASETS, METHODS, CONTRASTS, OLD_BATCH_RELATIVE
    from preflight_grouped_codec_fdma import plan as preflight_plan
    return dict(batch='grouped_codec_fdma_v1', fixed_old_batch=str(ROOT / OLD_BATCH_RELATIVE),
        training=[dict(arm=ARMS[0], seed=seed, total_updates=20000, included_smoke_updates=500,
                       best=fdma_config(seed).BEST_PATH) for seed in SEEDS],
        total_new_official_updates=40000, old_training_runs_repeated=0,
        preflight=preflight_plan(), evaluations={dataset: dict(test_seed=test_seed,
            training_seeds=list(SEEDS), gamma_db=[0,5,10,15,20,25], downlink_snr_db=25,
            batch_size=8, num_batches=200, samples_per_condition=1600, methods=list(METHODS),
            newly_forwarded_methods=[ARMS[0]] if dataset=='old_completion' else list(METHODS))
            for dataset,test_seed in DATASETS.items()},
        contrasts=CONTRASTS, primary_dataset='new_replication', primary_contrast='C_minus_F',
        primary_thresholds=dict(low_gamma_mean_min=.5,low_gamma_batch_ci95_lower_strictly_greater=0,
                                gamma25_batch_ci95_lower_min=-.1),
        retained_contrast='D_minus_E', datasets_pooled=False, training_seeds_pooled=False,
        initialization='random Legacy AS UE and common Joint BS; same existing seed addresses',
        fixed_energy_per_ue=64, occupied_re=256, noise_protocol='same existing absolute N0',
        historical_weights_used_for_training=False, cleanup=False, visits_08=False,
        adaptive_search=False, manuscript_modified=False)


def _environment():
    # Reuse accepted CUDA/Sionna/FP32 checks, not the old batch's execution stages.
    from run_grouped_codec_batch import _environment as accepted_environment
    result = accepted_environment()
    result['source'] = source_record()
    return result


def _training_record(seed):
    import torch
    from evaluate_grouped_codec_fdma import verify_training_metadata, build_fdma_model
    from train_grouped_codec_fdma import validate_model_state, validate_runtime_resume
    from evaluate_proposal_as_direct import check_tensor_state
    cfg = fdma_config(seed)
    frozen = {path: sha256(path) for path in (cfg.BEST_PATH, cfg.SAVE_PATH,
              cfg.CONFIG_PATH, cfg.STATUS_PATH, cfg.METRICS_PATH)}
    latest = torch.load(cfg.SAVE_PATH, map_location='cpu', weights_only=False)
    best = torch.load(cfg.BEST_PATH, map_location='cpu', weights_only=False)
    record = verify_training_metadata(seed, best, latest,
        dict(config=read_json(cfg.CONFIG_PATH), status=read_json(cfg.STATUS_PATH)))
    model, actual_cfg = build_fdma_model(torch, seed, best, torch.device('cpu'))
    validate_model_state(torch, model, latest['state'])
    check_tensor_state(torch, latest['state'], best['state'], equal=latest['step']==best['step'])
    validate_runtime_resume(torch, model, latest, actual_cfg, allow_completed=True)
    if any(sha256(path)!=digest for path,digest in frozen.items()):
        raise ValueError('F training evidence changed during completion inspection')
    return dict(arm=ARMS[0], training_seed=seed, completed_updates=latest['step'],
        best_step=best['step'], best_metric=best['best_metric'], best_path=cfg.BEST_PATH,
        latest_path=cfg.SAVE_PATH, best_sha256=frozen[cfg.BEST_PATH], latest_sha256=frozen[cfg.SAVE_PATH],
        metrics_path=cfg.METRICS_PATH, metrics_sha256=frozen[cfg.METRICS_PATH], verified_selection=record)


def _stage_child(folder, name, seed=None):
    output = folder / (name+'_'+uuid.uuid4().hex[:8]+'.json')
    command = [sys.executable, '-B', '-u', 'run_grouped_codec_fdma_batch.py',
               '--_stage', name, '--_output', str(output)]
    if seed is not None:
        command += ['--_seed', str(seed)]
    log = _child(folder, name, command)
    report = read_json(output)
    if report.get('passed') is not True:
        raise RuntimeError(f'Stage failed: {output}')
    evidence = [output, log]
    if name=='old_evidence':
        # Completed-stage reuse must rehash all pinned original files, not just this JSON.
        frozen = report.get('frozen_files')
        if not isinstance(frozen, dict) or not frozen:
            raise ValueError('Old evidence requires complete frozen file identities')
        for path,digest in frozen.items():
            path = safe_inside(path, ROOT)
            if sha256(path)!=digest:
                raise ValueError(f'Old evidence changed before stage completion: {path}')
            evidence.append(path)
    return dict(report=report, report_path=str(output), log=log, _evidence=evidence)


def _preflight(folder, old_evidence_path):
    output = folder / ('cuda_F_batch48_'+uuid.uuid4().hex[:8])
    retained = folder / (output.name+'.json')
    try:
        log = _child(folder, 'cuda_F_batch48', [sys.executable, '-B', '-u',
            'preflight_grouped_codec_fdma.py', '--run', '--output-dir', str(output),
            '--old-evidence', str(old_evidence_path)])
    finally:
        if (output/'report.json').is_file():
            save_json(retained, read_json(output/'report.json'))
    report = read_json(output/'report.json')
    if report.get('passed') is not True:
        raise RuntimeError('F-only CUDA acceptance failed; training forbidden')
    return dict(report=report, _evidence=[log, output/'report.json', retained])


def _train_one(folder, seed):
    cfg = fdma_config(seed)
    command = [sys.executable, '-B', '-u', 'train_grouped_codec_fdma.py', '--seed', str(seed)]
    logs = []
    if not Path(cfg.SAVE_PATH).exists():
        if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
            raise RuntimeError('F directory exists without latest; preserve and investigate')
        logs.append(_child(folder, f'legacy_fdma_{seed}_first500', command+['--stop-after','500']))
    if read_json(cfg.STATUS_PATH).get('state')!='completed':
        logs.append(_child(folder, f'legacy_fdma_{seed}_continue', command+['--resume','--train']))
    result = _stage_child(folder, 'training_record', seed)
    result['_evidence'] += logs+[cfg.BEST_PATH,cfg.SAVE_PATH,cfg.STATUS_PATH,cfg.CONFIG_PATH,cfg.METRICS_PATH]
    return result


def verify_evaluation(output, seed, dataset):
    from evaluate_grouped_codec_fdma import protocol, artifact_hashes, METHODS, DATASETS
    if dataset not in DATASETS:
        raise ValueError('Explicit old_completion or new_replication required')
    output = safe_inside(output, EVAL_ROOT/dataset)
    manifest = read_json(output/'manifest.json')
    expected_loaded = {ARMS[0]} if dataset=='old_completion' else set(METHODS)
    if (manifest.get('status')!='completed' or manifest.get('results_verified') is not True
            or manifest.get('freeze_checks',{}).get('passed') is not True
            or set(manifest.get('freeze_checks',{}).get('loaded_methods',[]))!=expected_loaded
            or manifest.get('shared_channel_noise_inputs_unchanged') is not True
            or manifest.get('samples_hashed_pairing') is not True
            or manifest.get('protocol')!=protocol(seed,dataset)):
        raise ValueError('F evaluation protocol/status/freeze/shared-input check failed')
    if (dataset=='old_completion' and (manifest.get('old_forward_calls')!=0
            or manifest.get('frozen_old_rows_preserved') is not True)):
        raise ValueError('Old completion must reuse all five immutable original rows without forwarding old models')
    inputs = manifest.get('input_sha256_after',{})
    code = manifest.get('runtime',{}).get('source_sha256',{})
    if not inputs or not code or inputs!=manifest.get('input_sha256_before'):
        raise ValueError('Missing or changed evaluation provenance')
    for name,digest in inputs.items():
        if sha256(safe_inside(name,ROOT))!=digest:
            raise ValueError(f'Changed evaluation input: {name}')
    for name,digest in code.items():
        if sha256(safe_inside(ROOT/name,ROOT))!=digest:
            raise ValueError(f'Changed evaluation source: {name}')
    if artifact_hashes(output)!=manifest.get('output_sha256'):
        raise ValueError('Changed evaluation artifacts')
    # Inputs are also stage evidence, so a later completed-stage skip cannot mask changes.
    return [output/'manifest.json']+[output/name for name in manifest['output_sha256']]+[
        safe_inside(name,ROOT) for name in inputs]+[safe_inside(ROOT/name,ROOT) for name in code]


def _evaluate_one(folder, seed, dataset):
    parent = safe_inside(EVAL_ROOT/dataset/folder.name/f'seed_{seed}',EVAL_ROOT/dataset)
    pointer = folder/f'evaluation_{dataset}_seed{seed}.json'
    if pointer.exists():
        previous = safe_inside(read_json(pointer)['directory'],parent)
        if (previous/'manifest.json').is_file() and read_json(previous/'manifest.json').get('status')=='completed':
            return dict(output_directory=str(previous),reused=True,
                        _evidence=verify_evaluation(previous,seed,dataset)+[pointer])
    output = parent/('attempt_'+uuid.uuid4().hex[:8])
    save_json(pointer,dict(directory=str(output)),replace=pointer.exists())
    retained = folder/f'evaluation_{dataset}_{seed}_{output.name}_manifest.json'
    try:
        log = _child(folder,f'evaluate_{dataset}_{seed}',[sys.executable,'-B','-u',
            'evaluate_grouped_codec_fdma.py','--seed',str(seed),'--dataset',dataset,
            '--evaluate','--output-dir',str(output)])
    finally:
        if (output/'manifest.json').is_file():
            save_json(retained,read_json(output/'manifest.json'))
    return dict(output_directory=str(output),log=log,
                _evidence=verify_evaluation(output,seed,dataset)+[log,pointer,retained])


def package_small(folder, summary):
    """Exclusive small allowlist ZIP; never archive weights, raw samples, logs or nested ZIPs."""
    folder = safe_inside(folder,BATCH_ROOT)
    prefix = 'results_bundle_' if summary['status']=='completed' else 'failure_bundle_'
    bundle = folder/(prefix+uuid.uuid4().hex+'.zip')
    temporary = bundle.with_name(bundle.name+'.partial')
    summary['return_files']=[str(folder/'batch_summary.json'),str(bundle)]
    save_json(folder/'batch_summary.json',summary,replace=True)
    selected = {folder/'batch_summary.json'}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in SMALL_SUFFIXES:
            selected.add(safe_inside(path,folder))
    for name,digest in summary.get('results',{}).get('files',{}).items():
        path=safe_inside(name,folder)
        if path.suffix.lower() not in SMALL_SUFFIXES:
            continue  # The result organizer's own ZIP must never be nested.
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError(f'Export changed before small packaging: {path}')
        selected.add(path)
    if any(path.stat().st_size>SMALL_FILE_LIMIT for path in selected):
        raise ValueError('A report exceeds the explicit small-file limit; preserve it without silently omitting evidence')
    if sum(path.stat().st_size for path in selected)>SMALL_TOTAL_LIMIT:
        raise ValueError('Small report package exceeds its fixed total limit')
    expected={path.relative_to(folder).as_posix():sha256(path) for path in sorted(selected)}
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            archive.write(path,path.relative_to(folder).as_posix())
    with zipfile.ZipFile(temporary,'r') as archive:
        names=archive.namelist()
        if len(names)!=len(expected) or set(names)!=set(expected):
            raise ValueError('Small ZIP member identity mismatch')
        for name,digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                raise ValueError('Small ZIP content mismatch')
    if any(sha256(folder/name)!=digest for name,digest in expected.items()):
        raise ValueError('Report changed while packaging')
    os.link(temporary,bundle)  # Exclusive publication, no overwrite.
    temporary.unlink()  # Only this invocation's verified temporary archive.
    summary.update(result_package=str(bundle),result_package_sha256=sha256(bundle),
        package_summary_note='The archive includes the pre-hash summary; the live summary records the finalized archive hash.')
    save_json(folder/'batch_summary.json',summary,replace=True)
    print(f'BATCH_SUMMARY={folder/"batch_summary.json"}',flush=True)
    print(f'RETURN_BUNDLE={bundle}',flush=True)
    return bundle


def run_batch(resume=False):
    if os.name!='posix':
        raise RuntimeError('Run the explicit batch on the CUDA server, not the preparation laptop')
    from evaluate_grouped_codec_fdma import DATASETS
    owner=SimpleNamespace(EXP_NAME='grouped_codec_fdma_batch_v1',arm='controller',SEED=0)
    with run_lock(owner,resume=resume,allowed_root=OUTPUT_ROOT):
        pointer=OUTPUT_ROOT/'last_batch.json'
        if resume:
            folder=safe_inside(read_json(pointer)['directory'],BATCH_ROOT)
            summary=read_json(folder/'batch_summary.json')
            if summary['source']['sha256']!=source_record()['sha256']:
                raise RuntimeError('Relevant source changed; refuse silent protocol changes')
        else:
            if pointer.exists():
                raise FileExistsError('Existing F batch; explicitly use --resume')
            for seed in SEEDS:
                cfg=fdma_config(seed)
                if Path(cfg.SAVE_DIR).exists() or Path(cfg.LOG_DIR).exists():
                    raise FileExistsError('Unowned F output; refuse overwrite')
            folder=safe_inside(BATCH_ROOT/('batch_'+uuid.uuid4().hex),BATCH_ROOT)
            folder.mkdir(parents=True,exist_ok=False)
            summary=dict(status='running',created=now(),source=source_record(),plan=plan(),
                stages={},cleanup='none',training={},historical_runs_retrained=False)
            save_json(folder/'batch_summary.json',summary)
            save_json(pointer,dict(directory=str(folder)))
        prior={key:summary.pop(key) for key in ('error','traceback','package_error','resume_command') if key in summary}
        if prior:
            summary.setdefault('failure_history',[]).append(dict(observed_at=now(),**prior))
        summary.pop('result_package',None)
        summary.pop('result_package_sha256',None)
        summary['return_files']=[str(folder/'batch_summary.json')]
        summary['status']='running'
        try:
            old_stage=_run_stage(summary,folder,'old_evidence',lambda:_stage_child(folder,'old_evidence'))
            old=old_stage['report']
            summary['old_evidence']=dict(report_path=old_stage['report_path'],
                summary_path=old['summary_path'],summary_sha256=old['summary_sha256'],
                frozen_file_count=len(old['frozen_files']))
            environment=_run_stage(summary,folder,'environment',lambda:_stage_child(folder,'environment'))['report']
            if resume:
                current=_stage_child(folder,'environment')['report']
                for key in ('torch','sionna','matplotlib','channel_backend','cuda'):
                    if current[key]!=environment[key]:
                        raise RuntimeError(f'Resume backend mismatch: {key}')
            summary['preflight']=_run_stage(summary,folder,'F_cuda_preflight',
                lambda:_preflight(folder,old_stage['report_path']))['report']
            training=deepcopy(old['training'])
            for seed in SEEDS:
                training.setdefault(str(seed),{})[ARMS[0]]=_run_stage(summary,folder,f'train_legacy_fdma_{seed}',
                    lambda seed=seed:_train_one(folder,seed))['report']
                summary['training']=training
            evaluations={dataset:{str(seed):_run_stage(summary,folder,f'evaluate_{dataset}_{seed}',
                lambda seed=seed,dataset=dataset:_evaluate_one(folder,seed,dataset))['output_directory']
                for seed in SEEDS} for dataset in DATASETS}
            summary['evaluations']=evaluations
            from grouped_codec_fdma_results import organize_results
            summary['results']=_run_stage(summary,folder,'organize_results',
                lambda:organize_results(folder,evaluations,training,old))
            # Rehash the old stage once more after all new work; old data remain read-only.
            _run_stage(summary,folder,'old_evidence',lambda:_stage_child(folder,'old_evidence'))
            summary.update(status='completed',finished=now())
        except BaseException as error:
            summary.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),
                resume_command='python -B -u run_grouped_codec_fdma_batch.py --run --resume')
            raise
        finally:
            try:
                package_small(folder,summary)
            except BaseException as error:
                summary.update(status='failed',package_error=f'{type(error).__name__}: {error}',
                    return_files=[str(folder/'batch_summary.json')])
                save_json(folder/'batch_summary.json',summary,replace=True)
                raise
    return folder


def _stat(path):
    path=Path(path)
    if not path.is_file():
        return dict(path=str(path),exists=False,mtime_utc=None,size=None)
    stat=path.stat()
    return dict(path=str(path),exists=True,size=stat.st_size,
        mtime_utc=datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat())


def training_progress(seed, *, cfg=None):
    """Live text-only observation, explicitly not a checkpoint-completion verifier."""
    cfg=fdma_config(seed) if cfg is None else cfg
    result=dict(arm=ARMS[0],seed=seed,checkpoint_verified=False,weights_read=False,
        latest_checkpoint_step=None,best_checkpoint_step=None,
        status=_stat(cfg.STATUS_PATH),metrics=_stat(cfg.METRICS_PATH),
        latest=_stat(cfg.SAVE_PATH),best=_stat(cfg.BEST_PATH),warnings=[])
    status={}
    if result['status']['exists']:
        try:
            status=read_json(cfg.STATUS_PATH)
            if not isinstance(status,dict) or status.get('experiment')!=cfg.EXP_NAME:
                raise ValueError('Status experiment differs')
        except (OSError,ValueError,TypeError) as error:
            result['warnings'].append('Status unreadable/inconsistent: '+str(error))
            status={}
    result.update(state=status.get('state','unknown' if result['latest']['exists'] else 'not_started'),
        status_step=status.get('step'),status_best_metric=status.get('best_metric'),
        metrics_last_step=None,metrics_last_train_step=None,metrics_last_validation_step=None,
        metrics_best_step=None,metrics_best_mean_all=None)
    if result['metrics']['exists']:
        try:
            with Path(cfg.METRICS_PATH).open(encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row=json.loads(line)
                        if not isinstance(row,dict):
                            raise ValueError('Metrics line must be an object')
                        step,metric=row.get('step'),row.get('selection_metric')
                        if type(step) is not int or not 0<step<=20000:
                            continue
                        if row.get('event')=='train_gamma' and step%100==0:
                            result['metrics_last_train_step']=step
                            result['metrics_last_step']=max(result['metrics_last_step'] or 0,step)
                            continue
                        if (step%500 or row.get('selection_metric_name')!='mean_all'
                                or isinstance(metric,bool) or not isinstance(metric,(int,float))
                                or not math.isfinite(metric)):
                            continue
                        result['metrics_last_validation_step']=step
                        result['metrics_last_step']=max(result['metrics_last_step'] or 0,step)
                        if result['metrics_best_mean_all'] is None or metric>result['metrics_best_mean_all']:
                            result['metrics_best_step']=step
                            result['metrics_best_mean_all']=metric
                    except (ValueError,TypeError):
                        result['warnings'].append('Ignored a partial/invalid live metrics line')
        except OSError as error:
            result['warnings'].append('Metrics unreadable: '+str(error))
    result['interpretation']='status and metrics only; unknown checkpoint-internal step remains null'
    return result


def _pairing_progress(folder,summary):
    """Read recorded manifest claims only, without silently rerunning hash/model checks."""
    from evaluate_grouped_codec_fdma import DATASETS
    result={}
    for dataset in DATASETS:
        result[dataset]={}
        for seed in SEEDS:
            key=f'evaluate_{dataset}_{seed}'
            row=dict(stage=summary.get('stages',{}).get(key,{}).get('status','unknown'),
                shared_input_pairing='unknown',old_rows_preserved='unknown',manifest_path=None,
                current_files_rehashed=False)
            pointer=folder/f'evaluation_{dataset}_seed{seed}.json'
            if pointer.is_file():
                try:
                    parent=EVAL_ROOT/dataset/folder.name/f'seed_{seed}'
                    output=safe_inside(read_json(pointer)['directory'],parent)
                    path=output/'manifest.json'
                    row['manifest_path']=str(path)
                    if path.is_file():
                        manifest=read_json(path)
                        row.update(manifest_status=manifest.get('status','unknown'),
                            shared_input_pairing=manifest.get('samples_hashed_pairing','unknown'),
                            shared_inputs_unchanged=manifest.get('shared_channel_noise_inputs_unchanged','unknown'),
                            old_rows_preserved=manifest.get('frozen_old_rows_preserved','unknown'),
                            recorded_batch_hashes=len(manifest.get('shared_batch_inputs',[])))
                except (OSError,ValueError,TypeError,KeyError,AttributeError) as error:
                    row['warning']='Evidence not readable: '+str(error)
            result[dataset][str(seed)]=row
    return result


def print_summary():
    pointer=OUTPUT_ROOT/'last_batch.json'
    if not pointer.exists():
        print('not_started: no experimental outcome inferred')
        return 0
    folder=safe_inside(read_json(pointer)['directory'],BATCH_ROOT)
    summary=read_json(folder/'batch_summary.json')
    stages=summary.get('stages',{})
    active=[key for key,value in stages.items() if value.get('status')=='running']
    if not active and summary.get('status')=='failed':
        active=[key for key,value in stages.items() if value.get('status')=='failed']
    old_stage=stages.get('old_evidence',{})
    initialization=summary.get('preflight',{}).get('initialization',{})
    print(json.dumps(dict(status=summary['status'],error=summary.get('error'),
        summary_path=str(folder/'batch_summary.json'),
        current_stage=active or ('none_completed' if summary['status']=='completed' else 'unknown'),
        latest_summary_record_utc=_stat(folder/'batch_summary.json')['mtime_utc'],
        stages={key:value['status'] for key,value in stages.items()},
        old_hash_acceptance=dict(recorded_stage_status=old_stage.get('status','unknown'),
            pinned_summary_sha256=summary.get('old_evidence',{}).get('summary_sha256'),
            frozen_file_count=summary.get('old_evidence',{}).get('frozen_file_count'),
            current_files_rehashed=False),
        initial_parameter_pairing={str(seed):initialization.get(str(seed),{}).get(
            'archived_initialization_matches','unknown') for seed in SEEDS},
        evaluation_pairing=_pairing_progress(folder,summary),
        result_package=summary.get('result_package'),result_package_sha256=summary.get('result_package_sha256'),
        return_files=summary.get('return_files',[])),indent=2))
    print('gamma: block-average transmit reference, not received SNR. CI: fixed-checkpoint sample and batch intervals; primary criteria use batch CI.')
    for seed in SEEDS:
        print('live_F '+json.dumps(training_progress(seed),sort_keys=True))
        record=summary.get('training',{}).get(str(seed),{}).get(ARMS[0])
        if record:
            print(f"verified_F seed={seed} latest={record['completed_updates']} best={record['best_step']} "
                  f"best_sha256={record['best_sha256']}")
    from evaluate_grouped_codec_fdma import METHODS,GAMMAS,DATASETS
    results=summary.get('results',{}).get('per_dataset',{})
    for dataset in DATASETS:
        for actual_seed in SEEDS:
            seed=str(actual_seed)
            result=results.get(dataset,{}).get(seed)
            if result is None:
                print(f'{dataset} seed={seed}: rates/paired intervals/C-F and D-E criteria unknown; evaluation/result audit not completed')
                continue
            print(f"{dataset} seed={seed} gamma={list(GAMMAS)} | C-F rule={result['engineering_criterion_C_minus_F']} "
                  f"| retained D-E rule={result['engineering_criterion_D_minus_E']}")
            for method in METHODS:
                rates=sorted((r for r in result['all_rates'] if r['method_key']==method),key=lambda r:float(r['gamma_db']))
                print(method+': '+' '.join(f"{float(r['sum_rate_mean']):.4f}" for r in rates))
            for name,comparison in result['comparisons'].items():
                print(name+' delta / sample95CI / batch95CI:')
                for label,row in [(str(g),r) for g,r in zip(GAMMAS,comparison['per_gamma'])]+[
                        ('low0/5/10',comparison['low_gamma']),('all6',comparison['all_gamma'])]:
                    print(f" {label}: {float(row['delta_mean']):+.4f} "
                          f"sample[{float(row['delta_ci95_low']):+.4f},{float(row['delta_ci95_high']):+.4f}] "
                          f"batch[{float(row['batch_delta_ci95_low']):+.4f},{float(row['batch_delta_ci95_high']):+.4f}]")
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    for name in ('dry-run','run','summary'):
        mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--_stage',choices=('old_evidence','environment','training_record'),help=argparse.SUPPRESS)
    parser.add_argument('--_output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--_seed',type=int,choices=SEEDS,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._stage:
        if args._output is None or args.run or args.resume or args.dry_run or args.summary:
            parser.error('Internal stage requires only explicit output and identity')
        if args._stage=='training_record' and args._seed is None:
            parser.error('Training record requires a seed')
        output=safe_inside(args._output,BATCH_ROOT)
        if output.exists():
            raise FileExistsError(output)
        try:
            if args._stage=='old_evidence':
                from evaluate_grouped_codec_fdma import verify_old_batch
                result=verify_old_batch()
            elif args._stage=='environment':
                result=_environment()
            else:
                result=_training_record(args._seed)
            save_json(output,dict(result,passed=True))
        except BaseException as error:
            if not output.exists():
                save_json(output,dict(passed=False,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()))
            raise
    elif args.resume and not args.run:
        parser.error('--resume requires --run')
    elif args.dry_run:
        print(json.dumps(plan(),indent=2))
    elif args.summary:
        return print_summary()
    elif args.run:
        run_batch(args.resume)
    else:
        parser.print_help()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
