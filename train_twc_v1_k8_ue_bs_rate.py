#!/usr/bin/env python3
"""Joint UE/BS rate updates with a causal historical receiver power prior.

Production requires the pinned CUDA sources. Explicit tiny CPU fixtures test
gradients, complete update boundaries and serialization without Sionna.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import traceback
import uuid

from configs import twc_v1_k8_ue_bs_rate_config as cfg
import ue_bs_rate_runtime as runtime
import train_twc_v1_k8_ul_adapt as replay

original, helpers = runtime.original, runtime.helpers
inside, atomic_json = runtime.inside, runtime.atomic_json
now, sha256, read_json = runtime.now, runtime.sha256, runtime.read_json
state_digest = runtime.state_digest
capture_rng, restore_rng = runtime.capture_rng, runtime.restore_rng
# These identify the deliberately replayed input stream, not this experiment.
TRACE_FIELDS, TRACE_FORMAT = replay.TRACE_FIELDS, replay.TRACE_FORMAT
initial_training_trace = replay.initial_training_trace
advance_training_trace = replay.advance_training_trace
validate_training_trace = replay.validate_training_trace
_canonical, _schema, _parameter_schema = replay._canonical, replay._schema, replay._parameter_schema


def _regular(path):
    path = inside(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular own joint-rate evidence required')
    return path


def _base_ema(source):
    value = source.get('base_ema_updates')
    if type(value) is not int or value < 0:
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
    """The legacy checker has the same one-group fresh AdamW recipe."""
    if (cfg.LR, cfg.WEIGHT_DECAY) != (replay.cfg.LR, replay.cfg.WEIGHT_DECAY):
        raise ValueError('Joint-rate optimizer no longer matches the locked recipe')
    replay.optimizer_check(optimizer, scheduler, step)


def _validate_power(stats, power_sum, frames, expected):
    import torch
    value = torch.as_tensor(power_sum, device=stats.power_ema.device).detach()
    if (not stats.training or type(frames) is not int or frames < 1
            or int(stats.ema_updates) != expected
            or value.shape != stats.power_ema.shape or value.dtype != stats.power_ema.dtype
            or not bool(torch.isfinite(value).all() & (value >= 0).all())
            or not bool(torch.isclose(value.sum(-1), torch.full_like(value.sum(-1), 48.*frames),
                atol=48.*frames*2e-5, rtol=1e-5).all())):
        raise ValueError('Complete actual transmitted-power sum and causal EMA boundary required')


def commit_update(codec, optimizer, scheduler, *, stats, power_sum, frames,
                  previous_updates, base_ema_updates):
    """Commit both trainable modules, then exactly one effective-batch EMA."""
    import torch
    optimizer_check(optimizer, scheduler, previous_updates)
    parameters = tuple(codec.parameters())
    actual = tuple(p for group in optimizer.param_groups for p in group['params'])
    if tuple(map(id, parameters)) != tuple(map(id, actual)):
        raise ValueError('Optimizer must contain exactly the UE and BS parameters')
    if not parameters or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError('Missing or nonfinite joint gradient; update refused')
    for module in (codec.encoder, codec.decoder):
        if not any(bool(p.grad.detach().abs().sum() > 0) for p in module.parameters()):
            raise ValueError('Rate loss must reach both UE encoder and BS decoder')
    _validate_power(stats, power_sum, frames, base_ema_updates+previous_updates)
    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError('Nonfinite joint parameter; EMA not committed')
    optimizer_check(optimizer, scheduler, previous_updates+1)
    stats.commit_power(power_sum, frames,
        expected_previous_updates=base_ema_updates+previous_updates)
    if int(stats.ema_updates) != base_ema_updates+previous_updates+1:
        raise ValueError('Exactly one historical EMA commit is required')
    return norm


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
    trace = validate_training_trace(trace, protocol, microbatches=step*cfg.ACCUMULATION)
    if production and step == cfg.UPDATES and trace != cfg.TRAINING_TRACE_PIN:
        raise ValueError('Joint input replay differs from the pinned BS-only training stream')
    return trace


def save_checkpoint(directory, codec, stats, ce, optimizer, scheduler, *, arm, step,
                    data, core, source, frozen, elapsed, measured, final=False,
                    production_evidence=True, training_trace=None):
    import torch
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
    trace = _trace_check(initial_training_trace(data.protocol) if training_trace is None else training_trace,
        data.protocol, step, production_evidence)
    initial_sha = sha256(_regular(directory/'initialization.json'))
    progress_sha = sha256(_regular(directory/'progress.jsonl')) if step == cfg.UPDATES else None
    payload = dict(format=cfg.NAMESPACE, arm=arm, method='joint', step=step,
        recipe=cfg.recipe(arm), fingerprint=cfg.fingerprint(arm), core=core,
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
    original.atomic_torch_save(payload, target)
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


def verify_saved(directory, payload, manifest, *, arm, data, core, source, frozen,
                 require_final=False, allow_fixture=False):
    directory = inside(directory)
    if directory != inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm)):
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
    identity = dict(format=cfg.NAMESPACE, arm=arm, method='joint', recipe=cfg.recipe(arm),
        fingerprint=cfg.fingerprint(arm), core=core, data_protocol=data.protocol,
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
            or manifest.get('recipe_fingerprint') != cfg.fingerprint(arm)
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


def verify_boundary(directory, arm, *, core=None):
    """Accept historical step100 independently of the overwritten latest file."""
    directory = inside(directory)
    record = read_json(_regular(directory/'boundary_00100.json'))
    initial_path = _regular(directory/'initialization.json')
    if (directory != inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm))
            or record.get('format') != cfg.NAMESPACE+'_manifest_v1'
            or record.get('arm') != arm or record.get('method') != 'joint'
            or record.get('step') != 100 or record.get('production_evidence') is not True
            or record.get('initialization_sha256') != sha256(initial_path)
            or record.get('fingerprint') != cfg.fingerprint(arm)
            or record.get('recipe') != cfg.recipe(arm)
            or record.get('counters') != _counters(100, record['source_identity'])
            or record.get('additional_EMA_commits') != 100):
        raise ValueError('Immutable joint100 identity changed')
    runtime.guard_core(record['core'])
    if core is not None and record['core'] != core:
        raise ValueError('Joint100 scientific code changed')
    runtime.verify_files(record['source_identity'])
    validate_training_trace(record['training_trace'], record['data_protocol'], microbatches=100*cfg.ACCUMULATION)
    return record


def verify_terminal_record(arm):
    """Recheck metadata and file bytes without allocating a model or CUDA."""
    directory = inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm))
    record = read_json(_regular(directory/'manifest.json'))
    source = record.get('source_identity', {})
    expected = dict(format=cfg.NAMESPACE+'_manifest_v1', status='completed', arm=arm,
        method='joint', step=cfg.UPDATES, counters=_counters(cfg.UPDATES, source),
        core=runtime.core_identity(), recipe=cfg.recipe(arm), fingerprint=cfg.fingerprint(arm),
        recipe_fingerprint=cfg.fingerprint(arm), production_evidence=True, evidence_kind='production',
        additional_EMA_commits=cfg.UPDATES, base_ema_updates=_base_ema(source),
        total_ema_updates=_base_ema(source)+cfg.UPDATES,
        data_protocol=runtime.make_training_data('cuda:0').protocol,
        initialization_sha256=sha256(_regular(directory/'initialization.json')))
    runtime.guard_core(expected['core'])
    if any(_canonical(record.get(k)) != _canonical(v) for k, v in expected.items()):
        raise ValueError('Exact immutable production joint terminal required')
    if (record != read_json(_regular(directory/'terminal_receipt.json'))
            or record.get('checkpoint_sha256') != sha256(_regular(directory/'checkpoint_latest.pt'))
            or record.get('progress_sha256') != sha256(_regular(directory/'progress.jsonl'))):
        raise ValueError('Joint terminal files changed')
    _trace_check(record['training_trace'], record['data_protocol'], cfg.UPDATES, True)
    initial = read_json(_regular(directory/'initialization.json'))
    for name in ('arm', 'method', 'recipe', 'fingerprint', 'core', 'data_protocol',
            'source_identity', 'frozen_identity', 'production_evidence', 'evidence_kind'):
        if initial.get(name) != record.get(name):
            raise ValueError('Joint terminal initialization changed')
    runtime.verify_files(source)
    return record


def load_finetuned(arm, device='cpu'):
    import torch
    directory = inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm))
    record = read_json(_regular(directory/'manifest.json'))
    if record.get('production_evidence') is not True:
        raise ValueError('CPU fixture cannot be loaded as production joint evidence')
    if torch.device(device).type != 'cuda':
        raise ValueError('Historical sources require the configured CUDA data identity')
    codec, stats, ce, source = runtime.load_initial(arm, 'joint', device)
    frozen = runtime.frozen_state(codec, ce, stats)
    path = _regular(directory/'checkpoint_latest.pt')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    verify_saved(directory, payload, record, arm=arm, data=runtime.make_training_data('cuda:0'),
        core=runtime.core_identity(), source=source, frozen=frozen, require_final=True)
    if (payload['model_schema'] != _schema(codec.state_dict())
            or payload['parameter_schema'] != _parameter_schema(codec)
            or payload['statistics_schema'] != _schema(stats.state_dict())):
        raise ValueError('Joint terminal architecture or receiver schema changed')
    original.strict_load(codec, payload['model'])
    original.strict_load(stats, payload['statistics'])
    runtime.assert_frozen_state(codec, ce, stats, frozen)
    for module in (codec, stats, ce):
        module.eval().requires_grad_(False)
    record = dict(record, path=str(path), sha256=sha256(path), state_sha256=state_digest(codec),
        CE_state_sha256=state_digest(ce), frozen_files=[*source['frozen_files'],
            *[dict(path=str(directory/name), sha256=sha256(directory/name)) for name in
                ('checkpoint_latest.pt', 'manifest.json', 'terminal_receipt.json', 'initialization.json', 'progress.jsonl')]],
        manifest_sha256=sha256(directory/'manifest.json'),
        terminal_receipt_sha256=sha256(directory/'terminal_receipt.json'),
        initialization_sha256=sha256(directory/'initialization.json'))
    return codec, stats, ce, record


def verify_terminal(arm):
    codec, stats, ce, record = load_finetuned(arm, runtime.configure_cuda())
    del codec, stats, ce
    return record


def monitor_if_missing(arm, step, codec, stats, ce, data, directory, *, forward=None,
                       production_evidence=True):
    if step not in cfg.MONITOR_STEPS:
        return
    import torch
    from evaluate_twc_v1_k8_icc_aligned import _tensor_hash, frozen_evaluation
    import evaluate_twc_v1_k8_ul_adapt as scoring
    import twc_v1_k8_icc_aligned_results as tables
    from physical_split_finish_io import write_union_csv
    forward = runtime.forward_codec if forward is None else forward
    directory = inside(directory)
    manifest = read_json(_regular(directory/'manifest.json'))
    receipt = directory/'monitor'/f'step_{step:05d}_receipt.json'
    expected = dict(arm=arm, method='joint', step=step, core=manifest['core'],
        recipe_fingerprint=cfg.fingerprint(arm), data_protocol=data.protocol,
        training_data_protocol=manifest['data_protocol'], checkpoint_sha256=manifest['checkpoint_sha256'],
        selection=False, production_evidence=production_evidence,
        evidence_kind='production' if production_evidence else 'test_fixture')
    if receipt.exists() or receipt.is_symlink():
        record = read_json(_regular(receipt))
        if any(record.get(k) != v for k, v in expected.items()) or not record.get('files'):
            raise ValueError('Joint monitor identity changed')
        for path, digest in record['files'].items():
            if sha256(_regular(path)) != digest:
                raise ValueError('Joint monitor bytes changed')
        return
    output = inside(directory/'monitor'/f'step_{step:05d}_{uuid.uuid4().hex[:10]}')
    output.mkdir(parents=True, exist_ok=False)
    before = (state_digest(codec), state_digest(stats), state_digest(ce))
    rows, fingerprints = [], []
    first, last = cfg.ROLE_RANGES['monitor']
    if last-first+1 != cfg.MONITOR_BATCHES:
        raise ValueError('Exact joint monitoring address budget required')
    with frozen_evaluation({'joint':(codec, stats)}, ce):
        for bi in range(cfg.MONITOR_BATCHES):
            index = first+bi
            sample = data.sample('validation', index)
            if sample.get('address') != data.address('validation', index):
                raise ValueError('Joint monitor address changed')
            for gi, gamma in enumerate(cfg.GAMMAS):
                out = forward(codec, stats, ce, sample, gamma_db=gamma,
                    noise_key=('monitor', bi, gi), method='joint')
                vectors = scoring.score_out(out, sample['H_dl'])
                scoring._validate_vectors(vectors, cfg.MICRO_BATCH)
                values = {k:v.detach().cpu().tolist() if torch.is_tensor(v) else list(v) for k,v in vectors.items()}
                for within in range(cfg.MICRO_BATCH):
                    rows.append(dict(arm='joint_'+arm, gamma_db=float(gamma), sample_index=bi*cfg.MICRO_BATCH+within,
                        batch_index=bi, within_batch=within, **{k:v[within] for k,v in values.items()}))
                fingerprints.append(dict(batch_index=bi, gamma_db=gamma, address=sample['address'],
                    H_dl=_tensor_hash(sample['H_dl']), H_ul=_tensor_hash(sample['H_ul']),
                    input_hashes=out.get('input_hashes')))
    if before != (state_digest(codec), state_digest(stats), state_digest(ce)):
        raise ValueError('Monitor changed joint model, historical prior or CE')
    summary = tables.summarize_rows(rows, samples=cfg.MONITOR_BATCHES*cfg.MICRO_BATCH,
        batch_size=cfg.MICRO_BATCH, arms=('joint_'+arm,))
    write_union_csv(output/'samples.csv.gz', rows)
    write_union_csv(output/'summary.csv', summary)
    atomic_json(output/'input_fingerprints.json', fingerprints, exclusive=True)
    files = {str(p):sha256(p) for p in output.iterdir() if p.is_file()}
    atomic_json(output/'manifest.json', dict(status='passed', phase='monitor', **expected,
        samples_per_condition=cfg.MONITOR_BATCHES*cfg.MICRO_BATCH, test_used=False, files=files), exclusive=True)
    files[str(output/'manifest.json')] = sha256(output/'manifest.json')
    atomic_json(receipt, dict(status='passed', **expected, files=files, path=str(output), updated=now()), exclusive=True)


def train(arm, stop_after=None, resume=False, *, require_cuda=True, loader=None,
          data_factory=None, forward=None, monitor=None):
    torch, np = original.tensor_runtime()
    target = cfg.UPDATES if stop_after is None else stop_after
    if (arm not in cfg.ARMS or type(target) is not int or not cfg.SAVE_EVERY <= target <= cfg.UPDATES
            or target % cfg.SAVE_EVERY):
        raise ValueError('Declared joint arm and complete fixed-budget boundary required')
    device = runtime.configure_cuda() if require_cuda else torch.device('cpu')
    core = runtime.core_identity()
    runtime.guard_core(core)
    random.seed(cfg.OPTIMIZER_SEED)
    np.random.seed(cfg.OPTIMIZER_SEED)
    torch.manual_seed(cfg.OPTIMIZER_SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.OPTIMIZER_SEED)
    fixture = not require_cuda or any(x is not None for x in (loader, data_factory, forward, monitor))
    loader = runtime.load_initial if loader is None else loader
    forward = runtime.forward_codec if forward is None else forward
    monitor = monitor_if_missing if monitor is None else monitor
    data_factory = runtime.make_data if data_factory is None else data_factory
    codec, stats, ce, source = loader(arm, 'joint', device)
    for name, module in (('encoder', codec.encoder), ('decoder', codec.decoder)):
        if state_digest(module) != source['base_'+name+'_state_sha256']:
            raise ValueError('Joint initialization differs from the matched source '+name)
    if state_digest(stats) != source['base_statistics_state_sha256'] or int(stats.ema_updates) != _base_ema(source):
        raise ValueError('Joint source prior state/counter changed')
    parameters = runtime.configure_joint(codec, ce, stats)
    frozen = runtime.frozen_state(codec, ce, stats)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        betas=(.9,.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
    data = data_factory(device, role='codec_train')
    monitor_data = data_factory(device, role='monitor')
    directory = inside(cfg.OUTPUT_ROOT/'training'/cfg.key(arm))
    if not resume:
        directory.mkdir(parents=True, exist_ok=False)
    elif not (directory/'checkpoint_latest.pt').is_file():
        raise FileNotFoundError('Own joint checkpoint required for explicit resume')
    start, trace = 0, initial_training_trace(data.protocol)
    with runtime.process_lock(directory/'training.lock'):
        if resume:
            manifest = read_json(_regular(directory/'manifest.json'))
            payload = torch.load(_regular(directory/'checkpoint_latest.pt'), map_location='cpu', weights_only=False)
            start = verify_saved(directory, payload, manifest, arm=arm, data=data, core=core,
                source=source, frozen=frozen, allow_fixture=fixture)
            if target < start:
                raise ValueError('Cannot rewind a joint checkpoint')
            if (payload['model_schema'] != _schema(codec.state_dict())
                    or payload['parameter_schema'] != _parameter_schema(codec)
                    or payload['statistics_schema'] != _schema(stats.state_dict())):
                raise ValueError('Joint resume architecture/schema changed')
            original.strict_load(codec, payload['model'])
            original.strict_load(stats, payload['statistics'])
            runtime.assert_frozen_state(codec, ce, stats, frozen)
            optimizer.load_state_dict(payload['optimizer'])
            scheduler.load_state_dict(payload['scheduler'])
            optimizer_check(optimizer, scheduler, start)
            restore_rng(payload['rng'])
            trace = dict(payload['training_trace'])
            ensure_boundary_receipts(directory, manifest)
        else:
            atomic_json(directory/'initialization.json', dict(arm=arm, method='joint', core=core,
                source_identity=source, frozen_identity=frozen, data_protocol=data.protocol,
                recipe=cfg.recipe(arm), fingerprint=cfg.fingerprint(arm),
                source_encoder_sha256=source['base_encoder_state_sha256'],
                source_decoder_sha256=source['base_decoder_state_sha256'],
                source_statistics_sha256=source['base_statistics_state_sha256'],
                initial_model_sha256=state_digest(codec), model_schema=_schema(codec.state_dict()),
                parameter_schema=_parameter_schema(codec), statistics_schema=_schema(stats.state_dict()),
                original_optimizer_continued=False, created=now(), production_evidence=not fixture,
                evidence_kind='production' if not fixture else 'test_fixture',
                training_trace_initial_sha256=trace['initial_sha256']), exclusive=True)
            save_checkpoint(directory, codec, stats, ce, optimizer, scheduler, arm=arm, step=0,
                data=data, core=core, source=source, frozen=frozen, elapsed=0., measured=0.,
                production_evidence=not fixture, training_trace=trace)
        started, timings, attempt = time.perf_counter(), [], uuid.uuid4().hex
        try:
            monitor(arm, start, codec, stats, ce, monitor_data, directory,
                forward=forward, production_evidence=not fixture)
            for step in range(start+1, target+1):
                begin = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                before_trace = dict(trace)
                prior = state_digest(stats)
                power_sum, losses, rates, fingerprints = None, [], [], []
                for micro in range(cfg.ACCUMULATION):
                    index = cfg.micro_index(step, micro)
                    sample = data.sample('codec_train', index)
                    if sample.get('address') != data.address('codec_train', index):
                        raise ValueError('Joint actual training address changed')
                    out = forward(codec, stats, ce, sample, gamma_db=None,
                        noise_key=('codec_train', index, 0), method='joint')
                    from evaluate_twc_v1_k8_icc_aligned import _tensor_hash
                    gamma = out.get('gamma_db')
                    expected = torch.tensor(cfg.gamma_schedule(index), dtype=torch.float32, device=device)
                    if (not torch.is_tensor(gamma) or gamma.requires_grad or not torch.equal(gamma, expected)
                            or out.get('input_hashes', {}).get('gamma') != _tensor_hash(gamma)):
                        raise ValueError('Joint actual gamma differs from the paired six-node schedule')
                    loss, metrics = helpers().rate_objective(out['H_hat'], sample['H_dl'],
                        alpha=cfg.ALPHA, downlink_snr_db=cfg.DL_SNR)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite complete-MU joint rate loss')
                    (loss/cfg.ACCUMULATION).backward()
                    actual_power = out['waveform'].active.detach().abs().square().sum(0)
                    power_sum = actual_power if power_sum is None else power_sum+actual_power
                    if state_digest(stats) != prior:
                        raise ValueError('Receiver prior mutated before the complete update')
                    losses.append(float(loss.detach()))
                    rates.append(float(metrics['sum_rate_bps_hz'].detach().mean()))
                    entry = dict(micro_index=index, address=sample['address'], channel=sample['sample_fingerprint'],
                        **out['input_hashes'])
                    trace = advance_training_trace(trace, entry, data.protocol)
                    fingerprints.append(entry)
                norm = commit_update(codec, optimizer, scheduler, stats=stats, power_sum=power_sum,
                    frames=cfg.MICRO_BATCH*cfg.ACCUMULATION, previous_updates=step-1,
                    base_ema_updates=_base_ema(source))
                runtime.assert_frozen_state(codec, ce, stats, frozen)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                timings.append(time.perf_counter()-begin)
                if step % cfg.SAVE_EVERY == 0:
                    measured = sum(timings[-100:])/len(timings[-100:])
                    row = dict(arm=arm, method='joint', step=step, attempt_id=attempt, resumed_from=start,
                        counters=_counters(step, source), loss=sum(losses)/cfg.ACCUMULATION,
                        sum_rate_bps_hz=sum(rates)/cfg.ACCUMULATION, gradient_norm=float(norm),
                        input_fingerprints=fingerprints, training_trace_before_update=before_trace,
                        training_trace=dict(trace), prior_state_before_update=prior,
                        prior_state_after_update=state_digest(stats), updated=now(),
                        mean_update_seconds=measured, ETA_seconds=measured*(cfg.UPDATES-step))
                    with (directory/'progress.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                    save_checkpoint(directory, codec, stats, ce, optimizer, scheduler, arm=arm, step=step,
                        data=data, core=core, source=source, frozen=frozen,
                        elapsed=time.perf_counter()-started, measured=measured, final=step == target,
                        production_evidence=not fixture, training_trace=trace)
                    print(f'{arm} joint UE+BS {step}/{cfg.UPDATES} rate={row["sum_rate_bps_hz"]:.6g} ETA={row["ETA_seconds"]:.0f}s', flush=True)
                monitor(arm, step, codec, stats, ce, monitor_data, directory,
                    forward=forward, production_evidence=not fixture)
            runtime.guard_core(core)
            runtime.verify_files(source)
            return read_json(directory/'manifest.json')
        except BaseException as exc:
            atomic_json(directory/'failure.json', dict(status='failed', arm=arm, step=locals().get('step', start),
                last_saved_step=read_json(directory/'manifest.json')['step'],
                error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), updated=now()))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--dry-run', action='store_true')
    parser.add_argument('--arm', choices=cfg.ARMS)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(root=str(cfg.OUTPUT_ROOT), recipe=cfg.recipe(args.arm or 'fdma'),
            weights_read=False, writes=False), indent=2))
        return 0
    if not args.run or args.arm is None:
        parser.error('Explicit --run --arm is required for joint training')
    train(args.arm, args.stop_after, args.resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
