#!/usr/bin/env python3
"""Hybrid-only direction-assisted joint updates with strict independent identity.

The historical FDMA model is never optimized by this module. Production uses
only the exact joint10000 Hybrid source. Small injected CPU fixtures exercise
serialization and replay and cannot become production terminal evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
import uuid

from configs import twc_v1_k8_hybrid_direction_config as cfg
import hybrid_direction_runtime as runtime
import twc_v1_k8_hybrid_direction as objective_math
import train_twc_v1_k8_ue_bs_rate as previous
import train_twc_v1_k8_ul_adapt as replay

original = previous.original
inside, atomic_json = runtime.inside, runtime.atomic_json
now, sha256, read_json = runtime.now, runtime.sha256, runtime.read_json
state_digest = runtime.state_digest
capture_rng, restore_rng = runtime.capture_rng, runtime.restore_rng
initial_training_trace, advance_training_trace = replay.initial_training_trace, replay.advance_training_trace
validate_training_trace = replay.validate_training_trace
_canonical, _schema, _parameter_schema = replay._canonical, replay._schema, replay._parameter_schema

def _regular(path):
    path = inside(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular own joint-rate evidence required')
    return path


def _base_ema(source):
    value = source.get('base_ema_updates')
    if type(value) is not int or value != 30000:
        raise ValueError('Exact source historical EMA counter required')
    return value


def _counters(step, source):
    return cfg.counters(step, base_ema_updates=_base_ema(source))


def _module_digest(state, prefix):
    selected = {name[len(prefix)+1:]:value for name,value in state.items()
        if name.startswith(prefix+'.')}
    if not selected:
        raise ValueError('Full joint checkpoint must contain both UE and BS states')
    return state_digest(selected)


def optimizer_check(optimizer, scheduler, step):
    if (cfg.LR, cfg.WEIGHT_DECAY, cfg.GRAD_CLIP) != (previous.cfg.LR, previous.cfg.WEIGHT_DECAY, previous.cfg.GRAD_CLIP):
        raise ValueError('Locked optimizer recipe changed')
    previous.optimizer_check(optimizer, scheduler, step)


def commit_update(*args, **kwargs):
    # The old helper is purely an optimizer/causal-EMA boundary operation.
    optimizer_check(args[1], args[2], kwargs['previous_updates'])
    return previous.commit_update(*args, **kwargs)


def _validate_statistics(payload):
    import torch
    state, schema = payload['statistics'], payload['statistics_schema']
    if (not isinstance(state, dict) or not state or not isinstance(schema, dict)
            or set(state) != set(schema) or not {'power_ema', 'ema_updates'} <= set(state)):
        raise ValueError('Complete receiver statistics and schema required')
    for name, value in state.items():
        if (not torch.is_tensor(value) or not bool(torch.isfinite(value).all())
                or schema[name] != dict(shape=list(value.shape), dtype=str(value.dtype))):
            raise ValueError('Joint receiver statistics shape/dtype/finiteness changed')
    counter = state['ema_updates']
    if (counter.dtype != torch.int64 or counter.ndim != 0
            or int(counter) != _base_ema(payload['source_identity'])+payload['step']):
        raise ValueError('Saved historical EMA counter changed')
    power = state['power_ema']
    if power.dtype != torch.float32 or power.ndim != 2 or not bool((power >= 0).all() & (power.sum(-1) > 0).all()):
        raise ValueError('Saved historical power prior is invalid')


def ensure_boundary_receipts(directory, manifest):
    directory = inside(directory)
    if (read_json(_regular(directory/'manifest.json')) != manifest
            or manifest['checkpoint_sha256'] != sha256(_regular(directory/'checkpoint_latest.pt'))):
        raise ValueError('Cannot issue a receipt for changed joint checkpoint bytes')
    names = []
    if manifest['step'] == 100:
        names.append('boundary_00100.json')
    if manifest['step'] == cfg.UPDATES:
        names.append('terminal_receipt.json')
    for name in names:
        path = directory/name
        if path.exists() or path.is_symlink():
            if read_json(_regular(path)) != manifest:
                raise ValueError('Immutable joint boundary receipt changed')
        else:
            atomic_json(path, manifest, exclusive=True)


def _trace_check(trace, protocol, step, production):
    # Fresh namespace: validate the exact new protocol/count, never compare to
    # the previous experiment's intentionally replayed trace endpoint.
    if production and protocol != runtime.make_data('cuda:0').protocol:
        raise ValueError('Production Hybrid training data protocol changed')
    return validate_training_trace(trace, protocol, microbatches=step*cfg.ACCUMULATION)


def save_checkpoint(directory, codec, stats, ce, optimizer, scheduler, *, arm=cfg.TRAIN_ARM, step,
                    data, core, source, frozen, elapsed, measured, final=False,
                    production_evidence=True, training_trace=None):
    import torch
    if arm != cfg.TRAIN_ARM:
        raise ValueError('Only Hybrid is trainable in this experiment')
    if type(step) is not int or not 0 <= step <= cfg.UPDATES or step % cfg.SAVE_EVERY:
        raise ValueError('Complete own joint save boundary required')
    runtime.guard_core(core)
    runtime.verify_files(source)
    optimizer_check(optimizer, scheduler, step)
    runtime.assert_frozen_state(codec, ce, stats, frozen)
    if int(stats.ema_updates) != _base_ema(source)+step:
        raise ValueError('Joint save requires one EMA commit per completed update')
    if type(production_evidence) is not bool:
        raise ValueError('Explicit production or CPU fixture identity required')
    if production_evidence and str(next(codec.parameters()).device) != 'cuda:0':
        raise ValueError('CPU fixture cannot be saved as production evidence')
    trace = _trace_check(initial_training_trace(data.protocol) if training_trace is None else training_trace,
        data.protocol, step, production_evidence)
    initial_sha = sha256(_regular(directory/'initialization.json'))
    progress_sha = sha256(_regular(directory/'progress.jsonl')) if step == cfg.UPDATES else None
    payload = dict(format=cfg.NAMESPACE, arm=arm, method='direction_joint', step=step,
        recipe=cfg.recipe(), fingerprint=cfg.fingerprint(), core=core,
        data_protocol=data.protocol, source_identity=source, frozen_identity=frozen,
        model=codec.state_dict(), model_schema=_schema(codec.state_dict()),
        parameter_schema=_parameter_schema(codec), statistics=stats.state_dict(),
        statistics_schema=_schema(stats.state_dict()), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), rng=capture_rng(), counters=_counters(step, source),
        training_code=runtime.source_identity(), initialization_sha256=initial_sha,
        production_evidence=production_evidence,
        evidence_kind='production' if production_evidence else 'test_fixture',
        training_trace=trace, progress_sha256=progress_sha)
    _validate_statistics(payload)
    target = inside(directory/'checkpoint_latest.pt')
    runtime.atomic_torch_save(payload, target)
    memory = (dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved()) if next(codec.parameters()).device.type == 'cuda'
        else dict(peak_allocated_bytes=0, peak_reserved_bytes=0))
    record = {k:payload[k] for k in ('arm', 'method', 'step', 'recipe', 'fingerprint', 'core',
        'data_protocol', 'source_identity', 'frozen_identity', 'counters', 'training_code',
        'initialization_sha256', 'production_evidence', 'evidence_kind', 'training_trace', 'progress_sha256')}
    record.update(format=cfg.NAMESPACE+'_manifest_v1',
        status='completed' if step == cfg.UPDATES else 'paused_boundary' if final else 'running',
        recipe_fingerprint=payload['fingerprint'], checkpoint_sha256=sha256(target),
        encoder_state_sha256=state_digest(codec.encoder), decoder_state_sha256=state_digest(codec.decoder),
        model_state_sha256=state_digest(codec), statistics_state_sha256=state_digest(stats),
        updated=now(), elapsed_seconds=elapsed, seconds_per_update=measured,
        ETA_seconds=measured*(cfg.UPDATES-step), memory=memory,
        trainable='UE encoder and BS decoder', additional_EMA_commits=step,
        base_ema_updates=_base_ema(source), total_ema_updates=int(stats.ema_updates))
    atomic_json(directory/'manifest.json', record)
    ensure_boundary_receipts(directory, record)
    return record


def verify_saved(directory, payload, manifest, *, arm=cfg.TRAIN_ARM, data, core, source, frozen,
                 require_final=False, allow_fixture=False):
    directory = inside(directory)
    if arm != cfg.TRAIN_ARM:
        raise ValueError('Only Hybrid is trainable in this experiment')
    if directory != inside(cfg.OUTPUT_ROOT/'training'/cfg.TRAIN_ARM):
        raise ValueError('Joint checkpoint arm/output changed')
    runtime.guard_core(core)
    runtime.verify_files(source)
    initial_path = _regular(directory/'initialization.json')
    required = {'format', 'arm', 'method', 'step', 'recipe', 'fingerprint', 'core', 'data_protocol',
        'source_identity', 'frozen_identity', 'model', 'model_schema', 'parameter_schema',
        'statistics', 'statistics_schema', 'optimizer', 'scheduler', 'rng', 'counters',
        'training_code', 'initialization_sha256', 'production_evidence', 'evidence_kind',
        'training_trace', 'progress_sha256'}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError('Exact complete joint checkpoint schema required')
    step = payload['step']
    if type(step) is not int or not 0 <= step <= cfg.UPDATES or step % cfg.SAVE_EVERY:
        raise ValueError('Complete joint update boundary required')
    identity = dict(format=cfg.NAMESPACE, arm=arm, method='direction_joint', recipe=cfg.recipe(),
        fingerprint=cfg.fingerprint(), core=core, data_protocol=data.protocol,
        source_identity=source, frozen_identity=frozen,
        initialization_sha256=sha256(initial_path), counters=_counters(step, source))
    for key, value in identity.items():
        if _canonical(payload.get(key)) != _canonical(value):
            raise ValueError('Joint resume identity changed: '+key)
    production = payload['production_evidence']
    if (type(production) is not bool or not allow_fixture and production is not True
            or payload['evidence_kind'] != ('production' if production else 'test_fixture')):
        raise ValueError('CPU fixture is not production joint training evidence')
    replay._validate_model_and_rng(payload)
    replay._validate_saved_optimizer(payload, step)
    _validate_statistics(payload)
    _trace_check(payload['training_trace'], data.protocol, step, production)
    progress_sha = sha256(_regular(directory/'progress.jsonl')) if step == cfg.UPDATES else None
    if payload['progress_sha256'] != progress_sha:
        raise ValueError('Terminal training journal changed')
    code = payload['training_code']
    if (not isinstance(code, dict) or set(code) != {'commit', 'sha256'}
            or code['sha256'] != core['sha256'] or not isinstance(code['commit'], str)
            or len(code['commit']) != 40 or any(c not in '0123456789abcdef' for c in code['commit'])):
        raise ValueError('Joint training source identity changed')
    if read_json(_regular(directory/'manifest.json')) != manifest:
        raise ValueError('Joint manifest content changed')
    for key in ('arm', 'method', 'step', 'recipe', 'fingerprint', 'core', 'data_protocol',
            'source_identity', 'frozen_identity', 'counters', 'training_code',
            'initialization_sha256', 'production_evidence', 'evidence_kind', 'training_trace', 'progress_sha256'):
        if _canonical(manifest.get(key)) != _canonical(payload[key]):
            raise ValueError('Joint manifest identity changed: '+key)
    if (manifest.get('format') != cfg.NAMESPACE+'_manifest_v1'
            or manifest.get('recipe_fingerprint') != cfg.fingerprint()
            or manifest.get('checkpoint_sha256') != sha256(_regular(directory/'checkpoint_latest.pt'))
            or manifest.get('additional_EMA_commits') != step
            or manifest.get('base_ema_updates') != _base_ema(source)
            or manifest.get('total_ema_updates') != _base_ema(source)+step
            or manifest.get('model_state_sha256') != state_digest(payload['model'])
            or manifest.get('encoder_state_sha256') != _module_digest(payload['model'], 'encoder')
            or manifest.get('decoder_state_sha256') != _module_digest(payload['model'], 'decoder')
            or manifest.get('statistics_state_sha256') != state_digest(payload['statistics'])
            or manifest.get('status') not in (('completed',) if step == cfg.UPDATES else ('running', 'paused_boundary'))):
        raise ValueError('Joint checkpoint bytes/state/counters/status changed')
    initial = read_json(initial_path)
    for key in ('arm', 'method', 'recipe', 'fingerprint', 'core', 'data_protocol',
            'source_identity', 'frozen_identity', 'production_evidence', 'evidence_kind'):
        if initial.get(key) != payload[key]:
            raise ValueError('Joint initialization identity changed: '+key)
    if (initial.get('original_optimizer_continued') is not False
            or initial.get('source_encoder_sha256') != source['base_encoder_state_sha256']
            or initial.get('source_decoder_sha256') != source['base_decoder_state_sha256']
            or initial.get('source_statistics_sha256') != source['base_statistics_state_sha256']
            or initial.get('model_schema') != payload['model_schema']
            or initial.get('parameter_schema') != payload['parameter_schema']
            or initial.get('statistics_schema') != payload['statistics_schema']
            or initial.get('training_trace_initial_sha256') != initial_training_trace(data.protocol)['initial_sha256']):
        raise ValueError('Joint initialization model/optimizer/prior/trace changed')
    if step == 0 and (initial.get('initial_model_sha256') != state_digest(payload['model'])
            or initial.get('source_statistics_sha256') != state_digest(payload['statistics'])):
        raise ValueError('Zero-update model or prior differs from its source')
    if require_final and (step != cfg.UPDATES or manifest['status'] != 'completed'
            or read_json(_regular(directory/'terminal_receipt.json')) != manifest):
        raise ValueError('Fixed joint terminal is incomplete')
    return step


def _directory():
    return inside(cfg.OUTPUT_ROOT/'training'/cfg.TRAIN_ARM)


def verify_boundary():
    """The immutable100 receipt remains checkable after latest advances."""
    directory = _directory()
    record = read_json(_regular(directory/'boundary_00100.json'))
    if (record.get('format') != cfg.NAMESPACE+'_manifest_v1'
            or record.get('arm') != cfg.TRAIN_ARM or record.get('method') != 'direction_joint'
            or record.get('step') != 100 or record.get('production_evidence') is not True
            or record.get('fingerprint') != cfg.fingerprint() or record.get('recipe') != cfg.recipe()
            or record.get('initialization_sha256') != sha256(_regular(directory/'initialization.json'))
            or record.get('counters') != _counters(100, record['source_identity'])
            or record.get('additional_EMA_commits') != 100):
        raise ValueError('Immutable Hybrid100 boundary identity changed')
    runtime.guard_core(record['core'])
    runtime.verify_source_record(cfg.TRAIN_ARM, record['source_identity'])
    runtime.verify_files(record['source_identity'])
    _trace_check(record['training_trace'], record['data_protocol'], 100, True)
    return record


def verify_terminal_record():
    """Metadata/file validation without allocating models or sampling data."""
    directory = _directory()
    record = read_json(_regular(directory/'manifest.json'))
    source = record.get('source_identity', {})
    expected = dict(format=cfg.NAMESPACE+'_manifest_v1', status='completed', arm=cfg.TRAIN_ARM,
        method='direction_joint', step=cfg.UPDATES, counters=_counters(cfg.UPDATES, source),
        core=runtime.core_identity(), recipe=cfg.recipe(), fingerprint=cfg.fingerprint(),
        recipe_fingerprint=cfg.fingerprint(), production_evidence=True, evidence_kind='production',
        additional_EMA_commits=cfg.UPDATES, base_ema_updates=30000,
        total_ema_updates=30000+cfg.UPDATES, data_protocol=runtime.make_data('cuda:0').protocol,
        initialization_sha256=sha256(_regular(directory/'initialization.json')))
    runtime.guard_core(expected['core'])
    if any(_canonical(record.get(k)) != _canonical(v) for k, v in expected.items()):
        raise ValueError('Exact completed production Hybrid terminal required')
    if (record != read_json(_regular(directory/'terminal_receipt.json'))
            or record.get('checkpoint_sha256') != sha256(_regular(directory/'checkpoint_latest.pt'))
            or record.get('progress_sha256') != sha256(_regular(directory/'progress.jsonl'))):
        raise ValueError('Hybrid terminal evidence bytes changed')
    _trace_check(record['training_trace'], record['data_protocol'], cfg.UPDATES, True)
    initial = read_json(_regular(directory/'initialization.json'))
    for key in ('arm', 'method', 'recipe', 'fingerprint', 'core', 'data_protocol',
                'source_identity', 'frozen_identity', 'production_evidence', 'evidence_kind'):
        if initial.get(key) != record.get(key):
            raise ValueError('Hybrid terminal initialization changed')
    runtime.verify_source_record(cfg.TRAIN_ARM, source)
    runtime.verify_files(source)
    return record


def load_finetuned(device='cpu'):
    import torch
    directory = _directory()
    record = read_json(_regular(directory/'manifest.json'))
    if record.get('production_evidence') is not True:
        raise ValueError('CPU fixture cannot become a production Hybrid terminal')
    if str(torch.device(device)) != 'cuda:0':
        raise ValueError('Production source requires canonical cuda:0 identity')
    verify_terminal_record()
    codec, stats, ce, source = runtime.load_source(cfg.TRAIN_ARM, device)
    frozen = runtime.frozen_state(codec, ce, stats)
    path = _regular(directory/'checkpoint_latest.pt')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    verify_saved(directory, payload, record, data=runtime.make_data('cuda:0'),
        core=runtime.core_identity(), source=source, frozen=frozen, require_final=True)
    if (payload['model_schema'] != _schema(codec.state_dict())
            or payload['parameter_schema'] != _parameter_schema(codec)
            or payload['statistics_schema'] != _schema(stats.state_dict())):
        raise ValueError('Hybrid terminal architecture/schema changed')
    original.strict_load(codec, payload['model']); original.strict_load(stats, payload['statistics'])
    runtime.assert_frozen_state(codec, ce, stats, frozen)
    for module in (codec, stats, ce): module.eval().requires_grad_(False)
    record = dict(record, path=str(path), sha256=sha256(path), state_sha256=state_digest(codec),
        CE_state_sha256=state_digest(ce), frozen_files=[*source['frozen_files'],
            *[dict(path=str(directory/name), sha256=sha256(directory/name)) for name in
                ('checkpoint_latest.pt', 'manifest.json', 'terminal_receipt.json', 'initialization.json', 'progress.jsonl')]],
        manifest_sha256=sha256(directory/'manifest.json'),
        terminal_receipt_sha256=sha256(directory/'terminal_receipt.json'))
    return codec, stats, ce, record


def train(stop_after=None, resume=False, *, require_cuda=True, loader=None, data_factory=None, forward=None):
    import torch
    import numpy as np
    target = cfg.UPDATES if stop_after is None else stop_after
    if (type(target) is not int or not cfg.SAVE_EVERY <= target <= cfg.UPDATES or target % cfg.SAVE_EVERY):
        raise ValueError('Complete fixed-budget Hybrid save boundary required')
    device = runtime.configure_cuda() if require_cuda else torch.device('cpu')
    core = runtime.core_identity(); runtime.guard_core(core)
    random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    if device.type == 'cuda': torch.cuda.manual_seed_all(cfg.SEED)
    fixture = not require_cuda or any(x is not None for x in (loader, data_factory, forward))
    loader = runtime.load_source if loader is None else loader
    data_factory = runtime.make_data if data_factory is None else data_factory
    forward = runtime.forward_codec if forward is None else forward
    codec, stats, ce, source = loader(cfg.TRAIN_ARM, device)
    runtime.verify_files(source)
    if not fixture: runtime.verify_source_record(cfg.TRAIN_ARM, source)
    for name, module in (('encoder', codec.encoder), ('decoder', codec.decoder)):
        if state_digest(module) != source['base_'+name+'_state_sha256']:
            raise ValueError('Hybrid initialization differs from audited source '+name)
    if state_digest(stats) != source['base_statistics_state_sha256'] or int(stats.ema_updates) != _base_ema(source):
        raise ValueError('Hybrid actual source prior/counter changed')
    parameters = runtime.configure_joint(codec, ce, stats)
    frozen = runtime.frozen_state(codec, ce, stats)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(.9,.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
    data, directory = data_factory(device), _directory()
    if not resume: directory.mkdir(parents=True, exist_ok=False)
    elif not (directory/'checkpoint_latest.pt').is_file():
        raise FileNotFoundError('Own saved Hybrid checkpoint required for explicit resume')
    start, trace = 0, initial_training_trace(data.protocol)
    with runtime.process_lock(directory/'training.lock'):
        if resume:
            manifest = read_json(_regular(directory/'manifest.json'))
            payload = torch.load(_regular(directory/'checkpoint_latest.pt'), map_location='cpu', weights_only=False)
            start = verify_saved(directory, payload, manifest, data=data, core=core,
                source=source, frozen=frozen, allow_fixture=fixture)
            if target < start: raise ValueError('Cannot rewind a Hybrid checkpoint')
            if (payload['model_schema'] != _schema(codec.state_dict())
                    or payload['parameter_schema'] != _parameter_schema(codec)
                    or payload['statistics_schema'] != _schema(stats.state_dict())):
                raise ValueError('Hybrid resume architecture/schema changed')
            original.strict_load(codec, payload['model']); original.strict_load(stats, payload['statistics'])
            runtime.assert_frozen_state(codec, ce, stats, frozen)
            optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
            optimizer_check(optimizer, scheduler, start); restore_rng(payload['rng'])
            trace = dict(payload['training_trace']); ensure_boundary_receipts(directory, manifest)
        else:
            atomic_json(directory/'initialization.json', dict(arm=cfg.TRAIN_ARM, method='direction_joint', core=core,
                source_identity=source, frozen_identity=frozen, data_protocol=data.protocol,
                recipe=cfg.recipe(), fingerprint=cfg.fingerprint(),
                source_encoder_sha256=source['base_encoder_state_sha256'],
                source_decoder_sha256=source['base_decoder_state_sha256'],
                source_statistics_sha256=source['base_statistics_state_sha256'],
                initial_model_sha256=state_digest(codec), model_schema=_schema(codec.state_dict()),
                parameter_schema=_parameter_schema(codec), statistics_schema=_schema(stats.state_dict()),
                original_optimizer_continued=False, created=now(), production_evidence=not fixture,
                evidence_kind='test_fixture' if fixture else 'production',
                training_trace_initial_sha256=trace['initial_sha256']), exclusive=True)
            save_checkpoint(directory, codec, stats, ce, optimizer, scheduler, step=0,
                data=data, core=core, source=source, frozen=frozen, elapsed=0., measured=0.,
                production_evidence=not fixture, training_trace=trace)
        started, timings, attempt = time.perf_counter(), [], uuid.uuid4().hex
        try:
            for step in range(start+1, target+1):
                begin = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                before_trace, prior = dict(trace), state_digest(stats)
                power_sum, logs, fingerprints = None, [], []
                for micro in range(cfg.ACCUMULATION):
                    index = cfg.micro_index(step, micro)
                    sample = data.sample('codec_train', index)
                    if sample.get('address') != data.address('codec_train', index):
                        raise ValueError('Hybrid training address changed')
                    out = forward(codec, stats, ce, sample, gamma_db=None, noise_key=('codec_train', index, 0))
                    from evaluate_twc_v1_k8_icc_aligned import _tensor_hash
                    gamma = out.get('gamma_db')
                    expected = torch.tensor(cfg.gamma_schedule(index), dtype=torch.float32, device=device)
                    if (not torch.is_tensor(gamma) or gamma.requires_grad or not torch.equal(gamma, expected)
                            or out.get('input_hashes', {}).get('gamma') != _tensor_hash(gamma)):
                        raise ValueError('Hybrid actual gamma differs from prescribed balanced nodes')
                    loss, metrics = objective_math.objective(out['H_hat'], sample['H_dl'])
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite Hybrid direction/rate loss')
                    (loss/cfg.ACCUMULATION).backward()
                    actual_power = out['waveform'].active.detach().abs().square().sum(0)
                    power_sum = actual_power if power_sum is None else power_sum+actual_power
                    if state_digest(stats) != prior: raise ValueError('Prior changed before complete effective update')
                    logs.append(dict(loss=float(loss.detach()),
                        sum_rate_bps_hz=float(metrics['sum_rate_bps_hz'].detach().mean()),
                        direction_error=float(metrics['direction_error'].detach().mean()),
                        weighted_direction_loss=float(metrics['weighted_direction_loss'].detach().mean())))
                    entry = dict(micro_index=index, address=sample['address'], channel=sample['sample_fingerprint'], **out['input_hashes'])
                    trace = advance_training_trace(trace, entry, data.protocol); fingerprints.append(entry)
                norm = commit_update(codec, optimizer, scheduler, stats=stats, power_sum=power_sum,
                    frames=cfg.MICRO_BATCH*cfg.ACCUMULATION, previous_updates=step-1, base_ema_updates=_base_ema(source))
                runtime.assert_frozen_state(codec, ce, stats, frozen)
                if device.type == 'cuda': torch.cuda.synchronize()
                timings.append(time.perf_counter()-begin)
                if step % cfg.SAVE_EVERY == 0:
                    measured = sum(timings[-100:])/len(timings[-100:])
                    row = dict(arm=cfg.TRAIN_ARM, method='direction_joint', step=step, attempt_id=attempt,
                        resumed_from=start, counters=_counters(step, source),
                        **{key:sum(log[key] for log in logs)/cfg.ACCUMULATION for key in logs[0]},
                        gradient_norm=float(norm), input_fingerprints=fingerprints,
                        training_trace_before_update=before_trace, training_trace=dict(trace),
                        prior_state_before_update=prior, prior_state_after_update=state_digest(stats),
                        gamma_counts=cfg.gamma_counts(step), updated=now(), mean_update_seconds=measured,
                        ETA_seconds=measured*(cfg.UPDATES-step))
                    with (directory/'progress.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                    save_checkpoint(directory, codec, stats, ce, optimizer, scheduler, step=step,
                        data=data, core=core, source=source, frozen=frozen, elapsed=time.perf_counter()-started,
                        measured=measured, final=step == target, production_evidence=not fixture, training_trace=trace)
                    print(f'Hybrid direction {step}/{cfg.UPDATES} rate={row["sum_rate_bps_hz"]:.6g} '
                        f'direction={row["direction_error"]:.6g} ETA={row["ETA_seconds"]:.0f}s', flush=True)
            runtime.guard_core(core); runtime.verify_files(source)
            return read_json(directory/'manifest.json')
        except BaseException as exc:
            atomic_json(directory/'failure.json', dict(status='failed', step=locals().get('step', start),
                last_saved_step=read_json(directory/'manifest.json')['step'], error=f'{type(exc).__name__}: {exc}',
                traceback=traceback.format_exc(), updated=now()))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store_true'); action.add_argument('--verify-terminal', action='store_true')
    parser.add_argument('--stop-after', type=int); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.verify_terminal:
        if args.stop_after is not None or args.resume: parser.error('Verification has no training cursor')
        print(json.dumps(verify_terminal_record(), indent=2, sort_keys=True))
    else: train(args.stop_after, args.resume)
    return 0


if __name__ == '__main__': raise SystemExit(main())
