#!/usr/bin/env python3
"""Independent six-method energy-gate evaluation; no historical result stitching."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

from configs.grouped_hybrid_power_train_config import (
    ARMS, SEEDS, OUTPUT_ROOT as BATCH_ROOT, SOURCE_PRIOR_OFFSET,
    power_config, source_identity, source_arm, channel_data_api, protocol as physical)
import evaluate_grouped_hybrid_anchor as anchor
from evaluate_proposal_as_direct import (
    sample_statistics, check_tensor_state, load_payload, read_json, write_manifest)
from evaluate_proposal_lite import _compare, model_buffer_snapshot, validate_frozen, validate_output_path
from evaluate_proposal_as_receiver import create_output, artifact_hashes as _artifact_hashes, runtime_record
from preflight_proposal_as_direct import sha256_file

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = BATCH_ROOT / 'evaluation'
TAG = 'k4_grouped_hybrid_power_v1'
TEST_SEED = 20270108
GAMMAS = (0, 5, 10, 15, 20, 25)
FROZEN = {'frozen_fdma_source': 'fdma_control', 'frozen_h32_source': 'h32_control'}
METHODS = (*ARMS, *FROZEN)
LAYOUTS = {m: ((64, 0) if m in ('fdma_control', 'frozen_fdma_source') else (32, 64)) for m in METHODS}
CONTRASTS = {
    'gamma_minus_H32': {'h32_gamma_power': 1, 'h32_control': -1},
    'static_minus_H32': {'h32_static_power': 1, 'h32_control': -1},
    'gamma_minus_static': {'h32_gamma_power': 1, 'h32_static_power': -1},
    'gamma_minus_FDMA': {'h32_gamma_power': 1, 'fdma_control': -1},
    'static_minus_FDMA': {'h32_static_power': 1, 'fdma_control': -1},
    'H32_minus_FDMA': {'h32_control': 1, 'fdma_control': -1},
    'FDMA_minus_frozen': {'fdma_control': 1, 'frozen_fdma_source': -1},
    'H32_minus_frozen': {'h32_control': 1, 'frozen_h32_source': -1},
}
PRIMARY = ('gamma_minus_FDMA', 'static_minus_FDMA', 'H32_minus_FDMA')
COMMON_METRICS = anchor.COMMON_METRICS + (
    'g_log_energy_ratio', 'exclusive_energy_per_active_symbol')
OPTIONAL_METRICS = anchor.OPTIONAL_METRICS + ('shared_energy_per_active_symbol',)
METRICS = COMMON_METRICS + OPTIONAL_METRICS
OUTPUT_ARTIFACTS = tuple(f'{name}_{TAG}.{suffix}' for name, suffix in (
    ('summary', 'csv'), ('samples', 'csv.gz'), ('paired_deltas', 'csv'),
    ('paired_samples', 'csv.gz'), ('parameters', 'csv'), ('gates', 'csv'),
    ('metadata', 'json'), ('results', 'json')))
SOURCE_FILES = tuple(dict.fromkeys(anchor.SOURCE_FILES + (
    'evaluate_grouped_hybrid_power.py', 'configs/grouped_hybrid_power_train_config.py',
    'train_grouped_hybrid_power.py', 'models/grouped_hybrid_power.py',
    'utils/grouped_hybrid_power.py')))


def make_test_data():
    return channel_data_api().ReplayableChannelData(
        base_seed=physical.namespace_seed(TEST_SEED, 'channel_test'), k_users=4, nt=32, nsc=256,
        batch_size=8, validation_batch_size=8, carrier_freq=3.5e9, speed=1., cdl_model='C', device='cuda:0')


def verify_test_namespace():
    current = make_test_data()
    seeds = {current.seed_for('training', i) for i in range(1, 201)}
    old = anchor.make_test_data()
    if current.protocol['data_id'] == old.protocol['data_id'] or seeds & {
            old.seed_for('training', i) for i in range(1, 201)}:
        raise ValueError('New power test namespace overlaps previous anchor test')
    for seed in SEEDS:
        cfg = power_config('fdma_control', seed)
        training = channel_data_api().ReplayableChannelData(
            base_seed=physical.namespace_seed(seed, 'channel'), k_users=4, nt=32, nsc=256,
            batch_size=cfg.BATCH_SIZE, validation_batch_size=cfg.VAL_BATCH,
            carrier_freq=cfg.CARRIER_FREQ, speed=cfg.SPEED, cdl_model='C', device='cuda:0')
        addresses = {training.seed_for('training', i) for i in range(1, 20001)}
        addresses.add(training.seed_for('validation'))
        if seeds & addresses:
            raise ValueError('New test namespace collides with training/validation')
    return dict(previous_anchor_test_disjoint=True, new_training_validation_disjoint=True,
                namespace='grouped_hybrid_power_v1', test_seed=TEST_SEED)


def protocol(training_seed, preflight=False):
    if type(training_seed) is not int or training_seed not in SEEDS or type(preflight) is not bool:
        raise ValueError('One prescribed training seed and explicit mode required')
    return dict(version='grouped-hybrid-power-evaluation-v1', training_seed=training_seed,
        seed=TEST_SEED, preflight=preflight, methods=list(METHODS), gamma_db=list(GAMMAS),
        downlink_snr_db=25, batch_size=8, num_batches=1 if preflight else 200,
        num_samples=8 if preflight else 1600, scenario=dict(K=4, Nt=32, Nsc=256, Dtot=256),
        physical_protocol=physical.protocol_metadata(), channel_data_protocol=make_test_data().protocol,
        namespace_check=verify_test_namespace(), channel_address='training(batch_index+1)',
        noise_address=dict(namespace='grouped_hybrid_power_v1', seed=TEST_SEED, split='test',
            index='zero-based batch_index', omits=['method', 'training_seed', 'gamma']),
        gamma_definition='10log10(64/(256*N0)); transmit reference, not historical received feedback SNR',
        noise_variance='N0=0.25*10**(-gamma/10); identical standard noise for all methods/gamma',
        layout_by_method={m: dict(Df_per_user=d, Da_per_group=a) for m, (d, a) in LAYOUTS.items()},
        gate_definition='g=ln(4)*tanh(theta/interpolated theta); exclusive amplitude exp(g/2), then joint energy64',
        prior_frozen=True, prior_count='source20000 + selected local step; frozen sources20000',
        rzf_regularization=1e-3, precoder_power_definition='sum over256 subcarriers; nominal256',
        numerical_policy=dict(real_dtype='float32', complex_dtype='complex64',
            matmul_precision='highest', allow_tf32=False, amp=False, microbatch=False),
        sample_ci_unit='complete_multi_user_channel', batch_ci_unit='sample_index//8',
        training_seeds_pooled=False, test_selects_checkpoint=False,
        ci_scope='Fixed-checkpoint sampling only; not training-seed stability',
        proxy_scope='Historical diagonal-prior linear error proxy, not calibrated posterior uncertainty',
        true_codewords='Detached offline diagnostics only; never decoder inputs',
        raw_nmse_scope='Secondary task-representation diagnostic, not a selection metric',
        perfect_csi_reference_included=False)


def checkpoint_plan(training_seed):
    if training_seed not in SEEDS:
        raise ValueError('Unknown training seed')
    result = []
    for arm in ARMS:
        cfg = power_config(arm, training_seed)
        result.append(dict(method_key=arm, best_path=cfg.BEST_PATH, latest_path=cfg.SAVE_PATH,
            config_path=cfg.CONFIG_PATH, status_path=cfg.STATUS_PATH,
            new_optimizer_updates=20000, source_identity=source_identity(arm, training_seed)))
    for method, arm in FROZEN.items():
        identity = source_identity(arm, training_seed)
        result.append(dict(method_key=method, new_optimizer_updates=0, source_identity=identity,
            **{key: identity[key] for key in ('best_path', 'latest_path', 'config_path', 'status_path')}))
    if len({r['best_path'] for r in result}) != 6:
        raise ValueError('Six distinct exact checkpoint paths required')
    return result


def sample_identity(method, gamma, index, training_seed):
    if method not in METHODS or gamma not in GAMMAS or type(index) is not int or index < 0 or training_seed not in SEEDS:
        raise ValueError('Invalid complete power result identity')
    d, a = LAYOUTS[method]
    return dict(method_key=method, training_seed=training_seed, seed=TEST_SEED, gamma_db=gamma,
        downlink_snr_db=25, Df_per_user=d, Da_per_group=a, occupied_feedback_RE=256,
        active_symbols_per_ue=d+a, sample_index=index, batch_index=index//8,
        noise_variance_N0=.25*10**(-gamma/10), diagnostic_schema='historical_power_prior_energy_gate_v1')


def _available(method, metric):
    return LAYOUTS[method][1] > 0 if metric in (
        'recovered_mse_shared', 'proxy_mse_shared', 'shared_energy_per_active_symbol') else True


def summarize_samples(samples, count, training_seed):
    if type(count) is not int or count < 8 or count % 8:
        raise ValueError('Complete batches of eight required')
    indexed = {(m, g): {} for m in METHODS for g in GAMMAS}
    for row in samples:
        key, index = (row.get('method_key'), row.get('gamma_db')), row.get('sample_index')
        if key not in indexed or type(index) is not int or not 0 <= index < count or index in indexed[key]:
            raise ValueError('Unexpected/duplicate full method-gamma-sample key')
        for name, value in sample_identity(*key, index, training_seed).items():
            _compare(value, row.get(name), 'Sample identity.'+name)
        if any(name in row for name in ('feedback_snr_db', 'allocation_df', 'allocation_da')):
            raise ValueError('Do not mix historical received-SNR/allocation protocols')
        for name in METRICS:
            if name not in row:
                raise ValueError('Missing explicit metric: '+name)
            if name in OPTIONAL_METRICS and not _available(key[0], name):
                if row[name] is not None:
                    raise ValueError('Empty shared support requires NA, not zero: '+name)
            elif row[name] is None or isinstance(row[name], bool) or not math.isfinite(float(row[name])):
                raise ValueError('Nonfinite metric: '+name)
        if any(abs(row[f'energy_ue{k}']-64)>1e-4 for k in range(4)):
            raise ValueError('Per-UE joint energy must remain64')
        ex, sh = row['exclusive_energy_mean_per_ue'], row['shared_energy_mean_per_ue']
        df, da = LAYOUTS[key[0]]
        if min(ex, sh)<0 or abs(ex+sh-64)>1e-4 or da == 0 and sh != 0:
            raise ValueError('Exclusive/shared energy accounting changed')
        if not math.isclose(row['exclusive_energy_per_active_symbol'], ex/df, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Exclusive energy per active symbol mismatch')
        if da and not math.isclose(row['shared_energy_per_active_symbol'], sh/da, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Shared energy per active symbol mismatch')
        if abs(row['g_log_energy_ratio']) > math.log(4)+1e-6:
            raise ValueError('Gate outside prescribed bounded log ratio')
        if key[0] not in ('h32_static_power', 'h32_gamma_power') and row['g_log_energy_ratio'] != 0:
            raise ValueError('Ungated reference cannot have a nonzero gate')
        if not 0 <= row['precoder_power_max_error'] <= 1e-5 or abs(row['precoder_power']-256)>256e-5:
            raise ValueError('Normalized RZF per-subcarrier power violated')
        nonnegative = [m for m in METRICS if ('_mse_' in m or '_linear_' in m or m.startswith('proxy_'))]
        nonnegative += ['empirical_noise_power', 'downlink_desired_mean', 'downlink_interference_mean']
        if any(row[m] is not None and row[m]<0 for m in nonnegative):
            raise ValueError('Negative squared-error/power diagnostic')
        indexed[key][index] = row
    result = []
    for (method, gamma), records in indexed.items():
        if set(records) != set(range(count)):
            raise ValueError('Missing complete paired sample IDs')
        row = sample_identity(method, gamma, 0, training_seed)
        row.pop('sample_index'); row.pop('batch_index')
        row.update(num_samples=count, num_batches=count//8, batch_size=8, preflight=count!=1600,
                   prior_frozen=True, rzf_regularization=1e-3)
        for metric in METRICS:
            values = [records[i][metric] for i in range(count)]
            stats = dict(mean=None, se=None, ci95_low=None, ci95_high=None) if values[0] is None else sample_statistics(values)
            row.update({metric+'_'+name: value for name, value in stats.items()})
        result.append(row)
    for gamma in GAMMAS:
        for i in range(count):
            reference = indexed[METHODS[0], gamma][i]['empirical_noise_power']
            if any(not math.isclose(indexed[m, gamma][i]['empirical_noise_power'], reference,
                                    rel_tol=1e-6, abs_tol=1e-10) for m in METHODS[1:]):
                raise ValueError('Different realized noise in supposedly paired observations')
    for method in METHODS:
        gate_values = {indexed[method, g][i]['g_log_energy_ratio'] for g in GAMMAS for i in range(count)}
        if method != 'h32_gamma_power' and len(gate_values) != 1:
            raise ValueError('Constant/absent gate changed across gamma or samples')
        for gamma in GAMMAS:
            if len({indexed[method, gamma][i]['g_log_energy_ratio'] for i in range(count)}) != 1:
                raise ValueError('Common public gamma gate cannot depend on individual test channels')
    return result, indexed


def paired_rows(summary, samples, count, training_seed):
    actual, indexed = summarize_samples(samples, count, training_seed)
    supplied = {(r.get('method_key'), r.get('gamma_db')): r for r in summary}
    if len(summary) != 36 or len(supplied) != 36:
        raise ValueError('Exactly36 unique summary conditions required')
    for row in actual:
        candidate = supplied.get((row['method_key'], row['gamma_db']), {})
        for name, value in row.items():
            if isinstance(value, float):
                if candidate.get(name) is None or not math.isfinite(float(candidate[name])) or not math.isclose(
                        value, float(candidate[name]), rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError('Summary/raw mismatch: '+name)
            else:
                _compare(value, candidate.get(name), 'Summary/raw.'+name)
    output, raw = [], []
    scopes = [('per_gamma', (g,)) for g in GAMMAS]+[
        ('low_gamma_mean', (0,5,10)), ('all_gamma_mean', GAMMAS)]
    for name, coefficients in CONTRASTS.items():
        for estimand, gammas in scopes:
            values = [statistics.fmean(sum(c*indexed[m,g][i]['sum_rate'] for m,c in coefficients.items())
                      for g in gammas) for i in range(count)]
            batches = [statistics.fmean(values[i:i+8]) for i in range(0,count,8)]
            row = dict(contrast_id=name, coefficients=json.dumps(coefficients, sort_keys=True),
                estimand=estimand, gamma_points=','.join(map(str,gammas)),
                gamma_db=gammas[0] if len(gammas)==1 else None, seed=TEST_SEED, training_seed=training_seed,
                downlink_snr_db=25, num_samples=count, num_batches=count//8, batch_size=8,
                sample_ci_unit='complete_multi_user_channel', batch_ci_unit='sample_index//8',
                batch_ci_estimable=count>8, training_seeds_pooled=False,
                role='primary' if name in PRIMARY and estimand=='low_gamma_mean' else
                     'secondary25' if name in PRIMARY and gammas==(25,) else 'diagnostic')
            for prefix, sequence in (('delta_', values), ('batch_delta_', batches)):
                row.update({prefix+k:v for k,v in sample_statistics(sequence).items()})
            output.append(row)
            raw.extend(dict(contrast_id=name, estimand=estimand, gamma_points=row['gamma_points'],
                sample_index=i, batch_index=i//8, delta=value, seed=TEST_SEED, training_seed=training_seed)
                for i,value in enumerate(values))
    return output, raw


def extract_diagnostics(torch, diagnostics, noise, gamma, method):
    old_arm = 'fdma_control' if LAYOUTS[method][1] == 0 else 'hybrid_f32_a64'
    values = anchor.extract_diagnostics(torch, diagnostics, noise, gamma, old_arm)
    df, da = LAYOUTS[method]
    batch = noise.shape[0]
    gate = diagnostics.get('g_log_energy_ratio')
    if method in FROZEN:
        gate = noise.real.new_zeros(())
    if not torch.is_tensor(gate) or gate.requires_grad or gate.numel()!=1 or not torch.isfinite(gate).all().item():
        raise ValueError('Detached scalar public-gamma log energy gate required')
    values.update(g_log_energy_ratio=gate.reshape(()).expand(batch),
        exclusive_energy_per_active_symbol=values['exclusive_energy_mean_per_ue']/df,
        shared_energy_per_active_symbol=None if da == 0 else values['shared_energy_mean_per_ue']/da)
    if method not in FROZEN:
        for name in ('exclusive_energy_per_active_symbol', 'shared_energy_per_active_symbol'):
            recorded = diagnostics.get(name)
            expected = values[name]
            if torch.is_tensor(recorded) and tuple(recorded.shape) == (batch, 4):
                recorded = recorded.mean(-1)
            if expected is None:
                if recorded is not None: raise ValueError('Empty shared support must be NA')
            elif not torch.is_tensor(recorded) or recorded.requires_grad or not torch.allclose(
                    recorded, expected, rtol=1e-6, atol=1e-6):
                raise ValueError('Model route energy diagnostic mismatch: '+name)
    return values


def artifact_hashes(output):
    return _artifact_hashes(output, expected_artifacts=OUTPUT_ARTIFACTS)


def verify_shared_inputs(manifest, reference=None):
    records = manifest.get('shared_batch_inputs', [])
    count = manifest.get('protocol', {}).get('num_batches')
    if count not in (1,200) or len(records) != count:
        raise ValueError('All actual shared test batch input hashes required')
    keys = {'batch_index','sample_index_start','sample_index_stop',
            'H_dl_sha256','H_ul_sha256','standard_noise_sha256'}
    for i, row in enumerate(records):
        if set(row)!=keys or row['batch_index']!=i or row['sample_index_start']!=8*i or row['sample_index_stop']!=8*(i+1):
            raise ValueError('Invalid/reordered batch input identity')
        for name in ('H_dl_sha256','H_ul_sha256','standard_noise_sha256'):
            value = row[name]
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid shared input SHA256')
    if reference is not None and records != reference:
        raise ValueError('Training seeds must share actual test channels/noise, not merely seed numbers')
    return records


def run_evaluation(training_seed, *, output_dir=None, preflight=False):
    p = protocol(training_seed, preflight)
    output = create_output(training_seed, output_dir=output_dir,
        root=OUTPUT_ROOT/('preflight' if preflight else 'formal'))
    manifest = dict(status='pending', protocol=p, methods={}, source_models={}, shared_batch_inputs=[],
        model_execution={}, prior_frozen=True, training_seeds_pooled=False, output_directory=str(output))
    hashes, snapshots, states, models, samples = {}, {}, {}, {}, []
    failure = None
    try:
        manifest['runtime'] = runtime_record(source_files=SOURCE_FILES)
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
        from train_grouped_hybrid_power import (
            load_source, verify_training_metadata, build_model, validate_model_state, apply_backend)
        from train_grouped_hybrid_anchor import build_model as build_frozen
        from preflight_joint_swin import tensor_digest
        import baseline_evaluate as ev
        manifest['runtime_before_backend'] = runtime_record(torch, source_files=SOURCE_FILES)
        apply_backend(torch)
        manifest['runtime'] = runtime_record(torch, source_files=SOURCE_FILES)
        device = torch.device('cuda:0'); ev.set_seed(TEST_SEED)
        source_payloads = {}
        for arm in ('fdma_control','h32_control'):
            payload, record = load_source(torch, arm, training_seed)
            source_payloads[arm] = payload; manifest['source_models'][arm] = record
            for name in ('best','latest','config','status'):
                path = record[name+'_path']; hashes[path] = sha256_file(path)
                _compare(record[name+'_sha256'], hashes[path], 'Verified frozen source identity')
        parameters, gates = [], []
        for entry in checkpoint_plan(training_seed):
            method = entry['method_key']
            if method in FROZEN:
                arm = FROZEN[method]; payload = source_payloads[arm]
                model, cfg = build_frozen(torch, source_arm(arm), training_seed, payload, device)
                record = dict(manifest['source_models'][arm], selected_best_step=20000,
                              prior_update_count=20000, new_optimizer_updates=0)
            else:
                latest = load_payload(torch, entry['latest_path'], hashes)
                best = load_payload(torch, entry['best_path'], hashes)
                sidecars = {n:read_json(entry[n+'_path'], hashes) for n in ('config','status')}
                record = verify_training_metadata(method, training_seed, best, latest, sidecars)
                model, cfg = build_model(torch, method, training_seed, best, device)
                validate_model_state(torch, model, latest['state'], step=latest['step'])
                check_tensor_state(torch, latest['state'], best['state'], equal=latest['step']==best['step'])
                record.update(selected_best_step=best['step'], prior_update_count=SOURCE_PRIOR_OFFSET+best['step'],
                              new_optimizer_updates=20000)
                del latest, best
            model.eval(); models[method] = (model,cfg)
            snapshots[method] = model_buffer_snapshot(model)
            validate_frozen(snapshots[method], snapshots[method])
            states[method] = tensor_digest(model.state_dict())
            record.update(best_path=entry['best_path'], latest_path=entry['latest_path'],
                best_sha256=hashes[entry['best_path']], latest_sha256=hashes[entry['latest_path']],
                strict_loading=True, model_class=type(model).__name__, architecture_metadata=model.architecture_metadata)
            manifest['methods'][method] = record
            gate_count=sum(v.numel() for name,v in model.named_parameters() if name.startswith('power_gate.'))
            expected_gate_count=1 if method=='h32_static_power' else 6 if method=='h32_gamma_power' else 0
            if gate_count!=expected_gate_count:
                raise ValueError('Actual gate parameter count differs from prescribed architecture')
            parameters.append(dict(method_key=method, training_seed=training_seed,
                full_parameters=sum(v.numel() for v in model.parameters()),
                ue_parameters=sum(v.numel() for v in model.ue_parameters()),
                bs_parameters=sum(v.numel() for v in model.bs_parameters()),
                gate_parameters=gate_count,
                note='Actual stored numel; UE includes gate; inactive Legacy branch included; not MACs/latency'))
        del source_payloads
        _compare(hashes, {path:sha256_file(path) for path in hashes}, 'Inputs changed during loading')
        data = make_test_data().prepare(); torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for batch_index in range(p['num_batches']):
                H_dl,H_ul = data.training(batch_index+1)
                noise = physical.standard_noise(TEST_SEED,'test',batch_index,8,device=str(device))
                originals = tuple(t.clone() for t in (H_dl,H_ul,noise))
                manifest['shared_batch_inputs'].append(dict(batch_index=batch_index,
                    sample_index_start=8*batch_index, sample_index_stop=8*(batch_index+1),
                    **{name+'_sha256':tensor_digest({name:value}) for name,value in
                       zip(('H_dl','H_ul','standard_noise'),originals)}))
                for gamma in GAMMAS:
                    for method,(model,cfg) in models.items():
                        result = model.forward_with_diagnostics(H_dl,H_ul,float(gamma),standard_noise=noise)
                        W,Hhat,d = result['precoder'],result['channel_surrogate'],result['diagnostics']
                        if not torch.isfinite(W).all().item() or not torch.isfinite(Hhat).all().item():
                            raise ValueError('Nonfinite deployment output')
                        if int(d['prior_update_count']) != manifest['methods'][method]['prior_update_count']:
                            raise ValueError('Frozen prior count differs from selected checkpoint cumulative count')
                        values = extract_diagnostics(torch,d,noise,gamma,method)
                        values.update(sum_rate=ev.per_sample_sum_rate(W,H_dl,10**(-25/10)),
                            nmse_db=ev.per_sample_nmse_db(Hhat,H_dl), precoder_power=ev.per_sample_precoder_power(W),
                            precoder_power_max_error=(W.abs().square().sum((1,2))-1).abs().amax(1),
                            **anchor.downlink_diagnostics(torch,W,H_dl))
                        arrays = {n:(None if v is None else v.detach().cpu().tolist()) for n,v in values.items()}
                        samples.extend(dict(sample_identity(method,gamma,8*batch_index+i,training_seed),
                            **{n:(None if v is None else v[i]) for n,v in arrays.items()}) for i in range(8))
                        manifest['model_execution'].setdefault(method,d['execution'])
                        if batch_index == 0:
                            theta = d.get('power_gate_theta')
                            gates.append(dict(method_key=method,training_seed=training_seed,gamma_db=gamma,
                                g_log_energy_ratio=arrays['g_log_energy_ratio'][0],
                                exclusive_to_shared_ratio_multiplier=math.exp(arrays['g_log_energy_ratio'][0]),
                                theta=json.dumps([] if theta is None else theta.detach().cpu().tolist()),
                                gate_depends_on_gamma=method=='h32_gamma_power'))
                if not all(torch.equal(a,b) for a,b in zip(originals,(H_dl,H_ul,noise))):
                    raise ValueError('Shared channels or standard noise modified in place')
        summary,_ = summarize_samples(samples,p['num_samples'],training_seed)
        paired,raw_pairs = paired_rows(summary,samples,p['num_samples'],training_seed)
        ev.write_csv(output/f'summary_{TAG}.csv',summary)
        ev.write_gzip_csv(output/f'samples_{TAG}.csv.gz',samples)
        ev.write_csv(output/f'paired_deltas_{TAG}.csv',paired)
        ev.write_gzip_csv(output/f'paired_samples_{TAG}.csv.gz',raw_pairs)
        ev.write_csv(output/f'parameters_{TAG}.csv',parameters);ev.write_csv(output/f'gates_{TAG}.csv',gates)
        for name,value in (('metadata',p),('results',dict(summary=summary,paired=paired))):
            with (output/f'{name}_{TAG}.json').open('x',encoding='utf-8') as stream:
                json.dump(value,stream,indent=2,allow_nan=False)
        manifest.update(status='completed',results_verified=True,summary_rows=36,paired_rows=64,
            shared_channel_noise_inputs_unchanged=True,samples_hashed_pairing=True,
            memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
        verify_shared_inputs(manifest)
    except Exception as error:
        failure=error;manifest.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        errors,after_hashes,after_buffers,after_states=[],{},{},{}
        for path,digest in hashes.items():
            try:
                after_hashes[path]=sha256_file(path)
                if after_hashes[path]!=digest: errors.append('Changed frozen input: '+path)
            except Exception as error: errors.append(str(error))
        for method,(model,cfg) in models.items():
            try:
                after_buffers[method]=model_buffer_snapshot(model)
                validate_frozen(snapshots[method],after_buffers[method])
                after_states[method]=tensor_digest(model.state_dict())
                if after_states[method]!=states[method]: raise ValueError('Model parameters/state changed')
                model.power_prior.validate_state(manifest['methods'][method]['prior_update_count'])
            except Exception as error: errors.append(method+':'+str(error))
        for name,digest in manifest.get('runtime',{}).get('source_sha256',{}).items():
            try:
                if sha256_file(ROOT/name)!=digest: errors.append('Source changed: '+name)
            except Exception as error: errors.append(str(error))
        manifest.update(input_sha256_before=hashes,input_sha256_after=after_hashes,
            model_buffers_before=snapshots,model_buffers_after=after_buffers,
            model_state_before=states,model_state_after=after_states,
            freeze_checks=dict(passed=not errors and len(models)==6,errors=errors,loaded_methods=list(models)))
        if errors or failure is None and len(models)!=6:
            failure=failure or RuntimeError('Frozen-state check failed');manifest.update(status='failed',error=str(failure))
        if failure is None:
            try: manifest['output_sha256']=artifact_hashes(output)
            except Exception as error: failure=error;manifest.update(status='failed',error=str(error))
        write_manifest(output,manifest)
    if failure is not None:
        raise RuntimeError(f'Power evaluation failed; retained evidence {output/"manifest.json"}') from failure
    print(f'OUTPUT_DIRECTORY={output}')
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    for mode in ('dry-run','preflight','evaluate'): modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    if not (args.dry_run or args.preflight or args.evaluate):parser.print_help();return 0
    if args.seed is None:parser.error('Explicit --seed required')
    if args.dry_run:
        if args.output_dir is not None:validate_output_path(args.output_dir.absolute(),OUTPUT_ROOT/'formal')
        plan=checkpoint_plan(args.seed)
        for entry in plan:
            source=entry['source_identity']
            configuration=source.pop('complete_source_configuration')
            source['complete_source_configuration_sha256']=hashlib.sha256(json.dumps(
                configuration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        print(json.dumps(dict(protocol=protocol(args.seed),checkpoints=plan,
            source_verified=False,checkpoint_bytes_not_read=True,models_not_imported=True,
            outputs_not_created=True),indent=2));return 0
    run_evaluation(args.seed,output_dir=args.output_dir,preflight=args.preflight)
    return 0


if __name__=='__main__':raise SystemExit(main())
