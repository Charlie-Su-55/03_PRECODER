#!/usr/bin/env python3
"""Frozen joint-terminal headroom and fixed-filter error diagnostics.

The default/plan/summary paths use only the standard library. Production --run
requires the historical strict CUDA loaders; CPU fixtures never become public
production evidence. No optimizer, source migration or remote publication runs
implicitly.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent
NAMESPACE = 'twc_v1_k8_rate_headroom_v1'
OUTPUT_ROOT = ROOT/'runs'/NAMESPACE
ARMS = ('fdma', 'hybrid_g4')
GAMMAS = (0, 5, 10, 15, 20, 25)
SEED, BATCH_SIZE, BATCHES, FIRST_INDEX = 20261126, 4, 64, 8192
SOURCE_CORE_SHA256 = '095de6af3e63a0d9bea38a7b59802043f71f7d616347e84269010da29caf58c5'
STUDY_FILES = ('diagnose_twc_v1_k8_rate_headroom.py',
    'twc_v1_k8_rate_headroom.py', 'publish_twc_v1_k8_rate_headroom.py',
    'publish_twc_v1_k8_ul_propagation.py',
    'twc_v1_k8_ul_propagation_metrics.py', 'twc_v1_k8_ul_diagnostics_metrics.py',
    'ul_diagnostics_runtime.py')
ID_FIELDS = frozenset(('arm', 'gamma_db', 'sample_index', 'batch_index', 'within_batch'))
RATE_METRICS = ('rate', 'true_csi_rzf_rate', 'true_direction_rzf_rate',
    'interference_free_upper_bound', 'true_csi_rzf_minus_current',
    'true_direction_rzf_minus_current', 'upper_bound_minus_current')
# These are audited joint10000 terminals, not the archived BS-only controls.
SOURCE_PINS = {
    "fdma": {
        "checkpoint_sha256": "ae82662fd9760275d34cd8ee6cc5763afe338560c456045c090270ebb572529e",
        "manifest_sha256": "99843ac3e3469272be421cbf71887fec446d190ef8f08029fe2b5335b7a95e3e",
        "terminal_receipt_sha256": "99843ac3e3469272be421cbf71887fec446d190ef8f08029fe2b5335b7a95e3e",
        "initialization_sha256": "f02a51cc965153128e1519193696c785fb63d22ff668bd39642604141c99bf54",
        "progress_sha256": "86fa4d1efb9b1c91c371c87ba3acb96c8e46e4bf96d76a41a46ba0dc2b018814",
        "encoder_state_sha256": "ad9ec60afb0fd09ffe6d73cfc74cdb5c15739a369879c82d074acb98b5c224fe",
        "decoder_state_sha256": "5c161dc446780860060b0a2b2c767fead15967838b563c3b3ba899d784f47e8f",
        "state_sha256": "4c9bc73bb66e036766829730e2f2d91cfa9f45dffd0adee50b8458ac4d574b89",
        "statistics_state_sha256": "f5ec8606acab789f116ce575474a4fa60b49f81d780a245f10e5e631e6dca4f8",
        "CE_state_sha256": "661b26bd6c041b7bfb9fa381177c51321114bb8beb6c3ea4b9f907b3e2ef8192"
    },
    "hybrid_g4": {
        "checkpoint_sha256": "6e5dc0de87135709aedbac67c89af3e4f08f374188117d2674158ed96ef589a6",
        "manifest_sha256": "6aacc722ff9829a03a2e28d04e46f07a0fd33d964116eb63230e81d4d262fb2c",
        "terminal_receipt_sha256": "6aacc722ff9829a03a2e28d04e46f07a0fd33d964116eb63230e81d4d262fb2c",
        "initialization_sha256": "d9bece55adf238e7b62becd1337d7635f081fe17f1171da1505d4f211110637b",
        "progress_sha256": "500cfb46d49befec5327948b5d73fc008e277035e31cab753672670ced4c696a",
        "encoder_state_sha256": "612c551ea3528cdf86d5adcbc7813b91ab0bafb1c5f45a2b3bd119c4b1b2aabc",
        "decoder_state_sha256": "31d3c8f896019869981cc3f117ad74155fac83897cd4a8b2739519837dc4df41",
        "state_sha256": "13bf6ebe9593ffed59d931c95ff45d1ac95969fffc6b18763375e7bbdbe357e5",
        "statistics_state_sha256": "66cd123da913626378920c37875709ba383fc9eedeadf926d953b7bc0ff0ff3a",
        "CE_state_sha256": "661b26bd6c041b7bfb9fa381177c51321114bb8beb6c3ea4b9f907b3e2ef8192"
    }
}



def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def plan():
    return dict(namespace=NAMESPACE, output=str(OUTPUT_ROOT), seed=SEED,
        data_namespace=NAMESPACE, role='validation', batch_size=BATCH_SIZE,
        batches=BATCHES, samples_per_arm_gamma=BATCH_SIZE*BATCHES,
        indices=[FIRST_INDEX, FIRST_INDEX+BATCHES-1], arms=ARMS, gamma_db=GAMMAS,
        optimizer_updates=0, UE_updates=0, BS_updates=0, CE_updates=0, EMA_updates=0,
        test_used=False, source_core_sha256=SOURCE_CORE_SHA256,
        references=['privileged true-CSI RZF(.03), not an upper bound',
            'privileged true direction with estimated row norms, not an upper bound',
            'interference-free waterfilling upper bound, potentially loose'],
        filter='unchanged estimated-UL receiver; truth only scores fixed filters',
        weights_read=False, writes=False)


def source_guard():
    from configs import twc_v1_k8_ue_bs_rate_config as cfg
    core = dict(schema=cfg.NAMESPACE+'_core_v1',
        sha256={name: sha(ROOT/name) for name in cfg.CORE_FILES})
    if digest(core) != SOURCE_CORE_SHA256:
        raise ValueError('Audited joint scientific core changed; no identity relaxation allowed')
    return core


def code_identity():
    core = source_guard()
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=ROOT, text=True).strip(), source_core=core,
        analysis_sha256={name: sha(ROOT/name) for name in STUDY_FILES})


def check_source_record(arm, record):
    if arm not in ARMS or any(record.get(k) != v for k, v in SOURCE_PINS[arm].items()):
        raise ValueError('Audited joint terminal file/state identity changed')
    if record.get('step') != 10000 or record.get('production_evidence') is not True:
        raise ValueError('Completed production joint10000 terminal required')


def atomic_json(path, value):
    path = Path(path)
    if (not path.resolve().is_relative_to(OUTPUT_ROOT.resolve()) or
            any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('Unlinked owned output path required')
    fd, name = tempfile.mkstemp(prefix='headroom_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def make_data(runtime, device):
    source = runtime.original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=SEED,
        batch_sizes=dict(ce_train=4, codec_train=4, ce_calibration=4, validation=4, test=1),
        device=str(device), cdl_model='C', speed=1., seed_namespace=NAMESPACE)


def batch_random(batch, gi, device):
    import torch
    def generator(kind):
        key = f'{NAMESPACE}:{SEED}:{FIRST_INDEX+batch}:{gi}:{kind}'.encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63-1)
        return torch.Generator(device=device).manual_seed(seed)
    noise = torch.view_as_complex(torch.randn(BATCH_SIZE, 32, 256, 2,
        generator=generator('noise'), device=device).contiguous())/2**.5
    polarity = (2*torch.randint(2, (BATCH_SIZE, 8),
        generator=generator('polarity'), device=device)-1).float()
    return noise, polarity


def frozen_signature(runtime, codec, stats, ce):
    for module in (codec, stats, ce):
        if any(m.training for m in module.modules()) or any(
                p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError('Diagnostic modules must be fully frozen in eval mode')
    return {name: runtime.state_digest(module) for name, module in
        (('codec', codec), ('encoder', codec.encoder), ('decoder', codec.decoder),
         ('statistics', stats), ('CE', ce))}


def score_batch(codec, stats, ce, sample, gamma, noise, polarity, *, runtime, raw=None):
    """Actual receiver first, then truth-only scoring; no state updates."""
    import torch
    import twc_v1_k8_rate_headroom as math_tools
    import twc_v1_k8_ul_propagation_metrics as propagation
    from evaluate_twc_v1_k8_icc_aligned import _tensor_hash
    phy = runtime.physics()
    dl = runtime.original.load_local('utils/twc_v1_k8_downlink.py', '_headroom_downlink')
    with torch.inference_mode():
        raw = codec.encoder(sample['H_dl']) if raw is None else raw
        wave = phy.build_waveform(raw, stats.layout, polarity=polarity)
        Y = phy.transmit(sample['H_ul'], wave.x_grid, noise, gamma)
        initial = phy.fourier_ls(Y, wave.pilot)
        estimated = ce(Y, wave.pilot, initial, gamma_db=gamma)
        cevar, prior = stats.ce_variance(gamma), stats.nominal_power()
        recovered = phy.recover(Y, estimated, stats.layout, prior, cevar, gamma)
        recovery_before = {k: _tensor_hash(recovered[k]) for k in
            ('soft', 'variance', 'reliability', 'H_ul_est', 'Y_res', 'prior')}
        out = codec.decoder(Y_rx=recovered['Y_res'], H_ul_est=recovered['H_ul_est'],
            soft=recovered['soft'], reliability=recovered['reliability'], pilots=wave.pilot,
            gamma_db=gamma, layout=stats.layout, prior=prior, ce_error=cevar)
        estimate = out if torch.is_tensor(out) else out['H_hat']
        observation = dict(Y=Y, estimated=estimated, prior=prior, wave=wave, ce_variance=cevar)
        truth = sample['H_dl']
        vectors = {}
        for name, H in (('rate', estimate), ('true_csi_rzf_rate', truth),
                ('true_direction_rzf_rate', math_tools.true_direction_reference(estimate, truth))):
            W, power = dl.design_rzf(H, .03)
            if not torch.allclose(power, torch.ones_like(power), atol=2e-6, rtol=2e-6):
                raise ValueError('Per-tone total precoder power changed')
            scored = dl.evaluate_sum_rate(W, truth, 25.)
            vectors[name] = scored['sum_rate_bps_hz'].double()
            if name == 'rate':
                vectors['desired_power'] = scored['desired_signal_power_mean'].double()
                vectors['interference_power'] = scored['interference_power_mean'].double()
        vectors['interference_free_upper_bound'] = math_tools.interference_free_upper_bound(truth)[
            'sum_rate_bps_hz'].double()
        if any(bool((vectors[n] > vectors['interference_free_upper_bound']+1e-3).any())
                for n in ('rate', 'true_csi_rzf_rate', 'true_direction_rzf_rate')):
            raise ValueError('Reference exceeds mathematical genie bound')
        for short, name in (('true_csi_rzf', 'true_csi_rzf_rate'),
                ('true_direction_rzf', 'true_direction_rzf_rate'),
                ('upper_bound', 'interference_free_upper_bound')):
            vectors[short+'_minus_current'] = vectors[name]-vectors['rate']
        vectors['csi_nmse_linear'] = dl.downlink_nmse(estimate, truth).mean(1).double()
        # This phase-invariant direction error does not mistake user phase for
        # an interference-sensitive error. It is descriptive, not a rate proof.
        dot = (estimate.conj()*truth).sum(2).abs().square().double()
        norms = estimate.abs().square().sum(2).double()*truth.abs().square().sum(2).double()
        vectors['csi_direction_sine_squared'] = (1-dot/norms.clamp_min(1e-30)).clamp(0, 1).mean((1, 2))
        decomposed = math_tools.receiver_error_decomposition(observation, recovered,
            sample['H_ul'], noise, stats.layout, gamma)
        # CE propagation decomposition uses the exact same A_est, paid Y,
        # historic prior and realized noise, including signed cross terms.
        ce_analysis = propagation.analyze(observation, recovered, sample['H_ul'], noise,
            stats.layout, gamma)
        nominal = prior.double()[None]
        for section, selected in decomposed['sections'].items():
            prefix = 'ul_'+section+'_'
            for name, value in decomposed['coordinate'].items():
                vectors[prefix+name] = (value[..., selected]/nominal[..., selected]).mean((1, 2))
            for name in ('realized_bias_power', 'realized_ce_power', 'realized_noise_power',
                    'realized_bias_ce_cross', 'realized_bias_noise_cross', 'realized_ce_noise_cross'):
                value = ce_analysis['coordinate'][name]
                vectors[prefix+name+'_ce_partition'] = (value[..., selected]/nominal[..., selected]).mean((1, 2))
        checks = dict(decomposed['checks'], **ce_analysis['checks'])
        filters = {k: _tensor_hash(v) for k, v in decomposed['filters'].items()}
        if filters != {k: _tensor_hash(v) for k, v in
                propagation.reconstruct_normal_filters(observation, recovered, stats.layout).items()}:
            raise ValueError('Offline truth scoring changed normal recovery filter')
        if recovery_before != {k: _tensor_hash(recovered[k]) for k in recovery_before}:
            raise ValueError('Offline scoring mutated normal receiver outputs')
        hashes = dict(H_dl=_tensor_hash(truth), H_ul=_tensor_hash(sample['H_ul']),
            standard_noise=_tensor_hash(noise), pilot_polarity=_tensor_hash(polarity),
            gamma=_tensor_hash(gamma), pilots=_tensor_hash(wave.pilot),
            estimated_UL=_tensor_hash(estimated), CE_variance=_tensor_hash(cevar),
            prior=_tensor_hash(prior), wave_active=_tensor_hash(wave.active), Y_rx=_tensor_hash(Y))
        if any(v.shape != (BATCH_SIZE,) or v.requires_grad or not bool(torch.isfinite(v).all())
                for v in vectors.values()):
            raise ValueError('Finite detached per-complete-MU scalar required')
        return vectors, hashes, filters, {k: float(v.max()) for k, v in checks.items()}


def verify_pairing(entries, *, batches=BATCHES):
    expected = {(a, b, float(g)) for a in ARMS for b in range(batches) for g in GAMMAS}
    index = {}
    common = ('H_dl', 'H_ul', 'standard_noise', 'pilot_polarity', 'gamma', 'pilots',
        'estimated_UL', 'CE_variance')
    for row in entries:
        key = row['arm'], row['batch_index'], float(row['gamma_db'])
        if key not in expected or key in index or row['address']['index'] != FIRST_INDEX+key[1]:
            raise ValueError('Unexpected or duplicate paired data address')
        if any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x)
                for x in (*row['fingerprint'].values(), *row['filters'].values())):
            raise ValueError('Actual tensor hashes required')
        index[key] = row
    if set(index) != expected: raise ValueError('Missing paired arm/gamma/batch')
    for b in range(batches):
        first = index['fdma', b, float(GAMMAS[0])]
        for g in GAMMAS:
            left, right = (index[a, b, float(g)] for a in ARMS)
            if left['address'] != right['address'] or any(
                    left['fingerprint'][k] != right['fingerprint'][k] for k in common):
                raise ValueError('Paired physical input bytes differ')
            if any(left['fingerprint'][k] != first['fingerprint'][k] for k in ('H_dl', 'H_ul')):
                raise ValueError('Gamma views used different physical channels')
    return index


def evaluate_models(runtime, loaded, data_factory, *, batches=BATCHES, progress=None):
    """Injectable real CPU path; production identity is enforced by run()."""
    import torch
    from evaluate_twc_v1_k8_icc_aligned import _tensor_hash
    rows, fingerprints, signatures, checks = [], [], {}, {}
    protocol = None
    with torch.inference_mode():
        for arm in ARMS:
            codec, stats, ce, record = loaded[arm]
            before = frozen_signature(runtime, codec, stats, ce)
            device = next(codec.parameters()).device
            data = data_factory(device)
            if protocol is not None and protocol != data.protocol:
                raise ValueError('Paired data protocol differs')
            protocol = data.protocol
            for bi in range(batches):
                sample = data.sample('validation', FIRST_INDEX+bi)
                if sample['address'] != data.address('validation', FIRST_INDEX+bi):
                    raise ValueError('Fresh validation data address changed')
                input_before = [_tensor_hash(sample[k]) for k in ('H_dl', 'H_ul')]
                raw = codec.encoder(sample['H_dl'])
                for gi, g in enumerate(GAMMAS):
                    gamma = torch.full((BATCH_SIZE,), float(g), device=device)
                    noise, polarity = batch_random(bi, gi, device)
                    values, hashes, filters, check = score_batch(codec, stats, ce, sample,
                        gamma, noise, polarity, runtime=runtime, raw=raw)
                    lists = {k: v.cpu().tolist() for k, v in values.items()}
                    for j in range(BATCH_SIZE):
                        rows.append(dict(arm=arm, gamma_db=float(g), batch_index=bi,
                            within_batch=j, sample_index=bi*BATCH_SIZE+j,
                            **{k: v[j] for k, v in lists.items()}))
                    fingerprints.append(dict(arm=arm, batch_index=bi, gamma_db=float(g),
                        address=sample['address'], fingerprint=hashes, filters=filters))
                    for k, v in check.items(): checks[k] = max(checks.get(k, 0.), v)
                if [_tensor_hash(sample[k]) for k in ('H_dl', 'H_ul')] != input_before:
                    raise ValueError('Offline reference mutated true channels')
                if progress: progress(arm, bi+1, batches)
            after = frozen_signature(runtime, codec, stats, ce)
            if before != after: raise ValueError('Frozen model/CE/prior changed')
            signatures[arm] = dict(before=digest(before), after=digest(after), states=before)
    verify_pairing(fingerprints, batches=batches)
    return rows, fingerprints, signatures, checks, protocol


def aggregate(rows):
    """Average gamma within MU, then batch four paired MUs before 95% CIs."""
    index = {}
    for row in rows:
        key = row['arm'], float(row['gamma_db']), row['sample_index']
        i = row['sample_index']
        if (key in index or key[0] not in ARMS or key[1] not in GAMMAS or
                type(i) is not int or i < 0 or row['batch_index'] != i//BATCH_SIZE or
                row['within_batch'] != i % BATCH_SIZE):
            raise ValueError('Duplicate or invalid complete-MU row')
        for k, v in row.items():
            if k not in ID_FIELDS and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
                raise ValueError('Finite scalar metric required')
        index[key] = row
    samples = len({k[2] for k in index})
    if samples < 2*BATCH_SIZE or samples % BATCH_SIZE or set(index) != {
            (a, float(g), i) for a in ARMS for g in GAMMAS for i in range(samples)}:
        raise ValueError('Complete paired six-gamma MU coverage and at least two batches required')
    for arm in ARMS:
        schema = set(index[arm, float(GAMMAS[0]), 0])-ID_FIELDS
        if not set(RATE_METRICS).issubset(schema) or any(
                set(index[arm, float(g), i])-ID_FIELDS != schema
                for g in GAMMAS for i in range(samples)):
            raise ValueError('Complete consistent metric schema required within each arm')
    scopes = [('point', (g,)) for g in GAMMAS]+[('low_0_5_10', GAMMAS[:3]), ('all_six', GAMMAS)]
    summaries, comparisons = [], []
    def moment(values):
        blocks = [statistics.mean(values[i:i+BATCH_SIZE]) for i in range(0, samples, BATCH_SIZE)]
        mean = statistics.mean(values); se = statistics.stdev(blocks)/len(blocks)**.5
        return dict(mean=mean, batch_se=se, ci_low=mean-1.96*se, ci_high=mean+1.96*se,
            n_samples=samples, n_batches=len(blocks))
    for scope, selected in scopes:
        common = dict(scope=scope, gamma_db=selected[0] if scope == 'point' else '',
            n_gamma_points=len(selected), batch_size=BATCH_SIZE)
        for arm in ARMS:
            metrics = set.intersection(*(set(index[arm, float(g), i])-ID_FIELDS
                for g in selected for i in range(samples)))
            for metric in sorted(metrics):
                values = [statistics.mean(index[arm, float(g), i][metric] for g in selected) for i in range(samples)]
                summaries.append(dict(common, arm=arm, metric=metric, **moment(values)))
        for metric in RATE_METRICS:
            values = [statistics.mean(index['hybrid_g4', float(g), i][metric]-
                index['fdma', float(g), i][metric] for g in selected) for i in range(samples)]
            comparisons.append(dict(common, contrast='hybrid_minus_fdma', metric=metric, **moment(values)))
    return summaries, comparisons


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(set(r) for r in rows))))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def verify_completed_export(path):
    import publish_twc_v1_k8_rate_headroom as publisher
    path = Path(path).resolve()
    manifest = publisher.verify_export(path)
    summary = json.loads((OUTPUT_ROOT/'summary.json').read_text())
    prereg = OUTPUT_ROOT/'preregistration.json'
    if (summary.get('status') != 'completed' or Path(summary['report_directory']).resolve() != path or
            summary.get('export_manifest_sha256') != sha(path/'export_manifest.json') or
            summary.get('preregistration_sha256') != sha(prereg)):
        raise ValueError('Completed local run/export binding changed')
    provenance = json.loads((path/'provenance.json').read_text())
    if provenance.get('preregistration_sha256') != sha(prereg):
        raise ValueError('Report preregistration binding changed')
    registered = json.loads(prereg.read_text())
    if (registered['plan'] != json.loads(canonical(plan())) or registered['code'] != code_identity()
            or registered['source_pins'] != SOURCE_PINS or registered['namespace'] != NAMESPACE):
        raise ValueError('Registered scientific code/recipe changed')
    public_plan = {k: v for k, v in plan().items() if k != 'output'}
    if (provenance.get('production_evidence') is not True or
            provenance.get('source_core_sha256') != SOURCE_CORE_SHA256 or
            provenance.get('plan') != json.loads(canonical(public_plan)) or
            provenance.get('code') != registered['code']):
        raise ValueError('Production report recipe/source provenance changed')
    import ue_bs_rate_runtime as runtime
    if provenance.get('data_protocol') != make_data(runtime, 'cuda:0').protocol:
        raise ValueError('Production validation data protocol changed')
    for arm in ARMS:
        check_source_record(arm, provenance['source_records'][arm])
        signature = provenance['frozen_signatures'][arm]
        states = signature['states']
        if signature['before'] != digest(states) or signature['after'] != signature['before']:
            raise ValueError('Frozen state signature changed')
        for state, field in (('codec', 'state_sha256'), ('encoder', 'encoder_state_sha256'),
                ('decoder', 'decoder_state_sha256'), ('statistics', 'statistics_state_sha256'),
                ('CE', 'CE_state_sha256')):
            if states.get(state) != SOURCE_PINS[arm][field]:
                raise ValueError('Public model state is not the pinned joint terminal')
    return manifest


def run(args):
    # Exclusive root creation prevents two controllers or silent reruns. All
    # failed evidence remains available; no cleanup touches historical roots.
    code = code_identity()
    if OUTPUT_ROOT.is_symlink() or any(p.is_symlink() for p in OUTPUT_ROOT.parents):
        raise ValueError('Linked diagnostic root forbidden')
    if OUTPUT_ROOT.exists():
        raise FileExistsError('Diagnostic root already exists; preserve and inspect its summary')
    # A busy GPU must not consume the fresh-run namespace. Repeat --run after
    # admission succeeds; source and scientific output remain untouched here.
    from run_twc_v1_k8_icc_aligned import gpu_admission
    admission = gpu_admission(allow_shared_gpu=args.allow_shared_gpu,
        min_free_mib=args.min_free_mib, max_gpu_util=args.max_gpu_util)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    prereg = dict(namespace=NAMESPACE, created=now(), plan=plan(), code=code, source_pins=SOURCE_PINS)
    atomic_json(OUTPUT_ROOT/'preregistration.json', prereg)
    summary = dict(namespace=NAMESPACE, status='starting', stage='startup_policy', updated=now(),
        commit=code['commit'], preregistration_sha256=sha(OUTPUT_ROOT/'preregistration.json'))
    def update(**values):
        summary.update(values, updated=now()); atomic_json(OUTPUT_ROOT/'summary.json', summary)
    update()
    try:
        atomic_json(OUTPUT_ROOT/'startup_policy.json', admission)
        import ue_bs_rate_runtime as runtime
        import train_twc_v1_k8_ue_bs_rate as trainer
        runtime.guard_core(code['source_core'])
        device = runtime.configure_cuda()
        loaded, records = {}, {}
        update(stage='strict_source_preflight')
        for arm in ARMS:
            codec, stats, ce, record = trainer.load_finetuned(arm, device)
            check_source_record(arm, record)
            actual = frozen_signature(runtime, codec, stats, ce)
            for state, field in (('codec', 'state_sha256'), ('encoder', 'encoder_state_sha256'),
                    ('decoder', 'decoder_state_sha256'), ('statistics', 'statistics_state_sha256'),
                    ('CE', 'CE_state_sha256')):
                if actual[state] != SOURCE_PINS[arm][field]:
                    raise ValueError('Loaded frozen state differs from audited endpoint')
            runtime.verify_files(record)
            loaded[arm] = codec, stats, ce, record
            records[arm] = dict(SOURCE_PINS[arm], step=10000, production_evidence=True)
        # No new final channel has been generated before both strict loads.
        update(stage='frozen_diagnostics')
        def progress(arm, complete, total):
            update(arm=arm, completed_batches=complete, batches_per_arm=total)
            print(f'{arm}: {complete}/{total} batches', flush=True)
        rows, inputs, signatures, checks, protocol = evaluate_models(runtime, loaded,
            lambda dev: make_data(runtime, dev), progress=progress)
        summaries, comparisons = aggregate(rows)
        for arm in ARMS: runtime.verify_files(loaded[arm][3])
        if code_identity() != code: raise ValueError('Scientific code changed during diagnostics')
        provenance = dict(namespace=NAMESPACE, format=NAMESPACE+'_provenance_v1',
            optimizer_updates=0, UE_updates=0, BS_updates=0, CE_updates=0, EMA_updates=0,
            test_used=False, contains_model_weights=False, contains_channel_tensors=False,
            source_core_sha256=SOURCE_CORE_SHA256, source_records=records,
            frozen_signatures=signatures, preregistration_sha256=summary['preregistration_sha256'],
            production_evidence=True, data_protocol=protocol,
            plan={k: v for k, v in plan().items() if k != 'output'}, code=code,
            paired_inputs_verified=True, reconstruction_max_errors=checks,
            true_UL_receiver_input=False, true_DL_receiver_input=False,
            privileged_references=True, source_selection='fixed_joint10000_terminals',
            interval='paired four-MU batch normal95; gamma averaged within MU; one training seed')
        import publish_twc_v1_k8_rate_headroom as publisher
        report = OUTPUT_ROOT/'reports'/('attempt_'+uuid.uuid4().hex[:12])
        publisher.create_export(report, dict(
            **{'README.md': ('Frozen joint-terminal diagnostics. No training updates; test sealed.\n'
                'True-CSI and true-direction RZF are privileged references, not upper bounds.\n'
                'The interference-free waterfilling bound may be loose. Signed UL decomposition\n'
                'powers use the frozen historical coordinate-prior scale; cross terms are retained.\n'
                'They do not identify a causal share of downstream rate loss.\n').encode()},
            **{'samples.csv': csv_bytes(rows), 'summary.csv': csv_bytes(summaries),
                'comparisons.csv': csv_bytes(comparisons), 'input_fingerprints.json': canonical(inputs),
                'provenance.json': canonical(provenance)}))
        update(status='completed', stage='directory_export', report_directory=str(report),
            export_manifest_sha256=sha(report/'export_manifest.json'))
        verify_completed_export(report)
    except BaseException as exc:
        update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise


def compact_summary():
    path = OUTPUT_ROOT/'summary.json'
    if not path.exists(): return 'Rate headroom | not started | zero training updates'
    summary = json.loads(path.read_text())
    lines = [f"Rate headroom | {summary['status']} | stage={summary['stage']}",
        'Namespace: '+NAMESPACE, 'Updated: '+summary['updated'], 'Commit: '+summary['commit'],
        'Frozen joint10000 FDMA/Hybrid; optimizer/UE/BS/CE/EMA updates=0; test sealed']
    if 'completed_batches' in summary:
        lines.append(f"Progress: {summary['arm']} {summary['completed_batches']}/{BATCHES} batches")
    if summary['status'] == 'completed':
        report = Path(summary['report_directory']); verify_completed_export(report)
        with (report/'summary.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['scope'] == 'all_six' and row['metric'] in RATE_METRICS:
                    lines.append(f"{row['arm']} {row['metric']}: {float(row['mean']):.6f} "
                        f"batchCI=[{float(row['ci_low']):.6f},{float(row['ci_high']):.6f}]")
        lines.append('Report directory: '+str(report))
        lines.append('Upload: '+sys.executable+' -B '+str(ROOT/'publish_twc_v1_k8_rate_headroom.py')+
            ' --publish --export-dir '+str(report))
    if 'error' in summary: lines.append('ERROR: '+summary['error'])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--plan', action='store_true'); action.add_argument('--run', action='store_true')
    action.add_argument('--summary', action='store_true')
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--min-free-mib', type=int, default=12000)
    parser.add_argument('--max-gpu-util', type=int, default=20)
    args = parser.parse_args(argv)
    if args.run: run(args); print(compact_summary())
    elif args.summary: print(compact_summary())
    else: print(json.dumps(plan(), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
