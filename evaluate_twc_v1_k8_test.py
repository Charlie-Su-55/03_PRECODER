#!/usr/bin/env python3
"""ONE-TIME sealed test evaluation of the K8 paper models (forward passes only).

Everything is fixed before this run: frozen FDMA, terminal final Hybrid, terminal
pairing-aware fine-tune, the training-free baselines, and the per-gamma switch decided
on the selection set in twc_v1_k8_paired_ft_v1/reports/attempt_180e3d7891b1. Nothing is
selected or tuned on test. The run refuses to start if a test report already exists.

Test frames: role 'test' of the K8 replayable data (isolated index range), 256 batches
of 4 complete MUs; feedback noise, sounding noise and pilot polarity from a test-only
seed namespace. Same six gamma nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid

from configs import twc_v1_k8_final_config as cfg
import final_runtime as rt
import train_twc_v1_k8_final as trainer
import train_twc_v1_k8_paired_ft as ft
import evaluate_twc_v1_k8_final as base_eval
import evaluate_twc_v1_k8_extended as ext
import evaluate_twc_v1_k8_baselines as bl
import evaluate_twc_v1_k8_sscc as sc
import twc_v1_k8_final_results as results

NAMESPACE = 'twc_v1_k8_test_v1'
OUTPUT_ROOT = Path('runs')/NAMESPACE
SWITCH_SOURCE = Path('runs/twc_v1_k8_paired_ft_v1/reports/attempt_180e3d7891b1/provenance.json')
TEST_BATCHES = 256
NETWORK = dict(
    fdma_baseline=('fdma', {}), fdma_genie_ul=('fdma', dict(genie_ul=True)),
    hybrid_final=('final', {}), hybrid_paired=('final', dict(pairing='least', sounding=True)),
    hybrid_random_pairing=('final', dict(pairing='random', sounding=True)),
    hybrid_most_coupled=('final', dict(pairing='most', sounding=True)),
    hybrid_paired_genie_ul=('final', dict(pairing='least', sounding=True, genie_ul=True)),
    ft_paired=('ft', dict(pairing='least', sounding=True)))
OTHER = ('analog_fdma', 'analog_hybrid_paired', 'sscc_fdma', 'perfect_csi_rzf')
METRICS = base_eval.METRICS+tuple(f'sum_rate_dl{s}' for s in ext.DL_SNRS)
CONTRASTS = (('proposed_minus_fdma', 'proposed', 'fdma_baseline'),
             ('ft_paired_minus_fdma', 'ft_paired', 'fdma_baseline'),
             ('paired_minus_fdma', 'hybrid_paired', 'fdma_baseline'),
             ('final_minus_fdma', 'hybrid_final', 'fdma_baseline'),
             ('ft_minus_paired', 'ft_paired', 'hybrid_paired'),
             ('paired_minus_random', 'hybrid_paired', 'hybrid_random_pairing'),
             ('paired_minus_most_coupled', 'hybrid_paired', 'hybrid_most_coupled'),
             ('genie_minus_paired', 'hybrid_paired_genie_ul', 'hybrid_paired'),
             ('fdma_genie_minus_fdma', 'fdma_genie_ul', 'fdma_baseline'),
             ('genie_paired_minus_genie_fdma', 'hybrid_paired_genie_ul', 'fdma_genie_ul'),
             ('upper_minus_proposed', 'perfect_csi_rzf', 'proposed'),
             ('upper_minus_fdma', 'perfect_csi_rzf', 'fdma_baseline'),
             ('analog_paired_minus_analog_fdma', 'analog_hybrid_paired', 'analog_fdma'),
             ('fdma_minus_sscc', 'fdma_baseline', 'sscc_fdma'),
             ('sscc_minus_analog', 'sscc_fdma', 'analog_fdma'))


def seed(index, kind):
    raw = f'{NAMESPACE}:{cfg.SEED}:test:{index}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-1)


def streams(index, device):
    import torch
    def cnoise(kind):
        g = torch.Generator(device=device).manual_seed(seed(index, kind))
        return torch.view_as_complex(torch.randn(cfg.MICRO_BATCH, rt.NT, rt.NSC, 2, generator=g,
            device=device).contiguous())/2**.5
    g = torch.Generator(device=device).manual_seed(seed(index, 'polarity'))
    polarity = (2*torch.randint(2, (cfg.MICRO_BATCH, rt.K), generator=g, device=device)-1).float()
    return cnoise('noise'), cnoise('sounding'), polarity


def test_data(device):
    source = rt.base().original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.SEED,
        batch_sizes=dict(ce_train=4, codec_train=cfg.MICRO_BATCH, ce_calibration=4,
            validation=cfg.MICRO_BATCH, test=cfg.MICRO_BATCH), device=str(device), cdl_model='C', speed=1.,
        seed_namespace=cfg.NAMESPACE)


def switch_decision(path=SWITCH_SOURCE):
    decision = json.loads(Path(path).read_text())['switch_decision']
    mapping = {'fdma_baseline': 'fdma_baseline', 'hybrid_paired': 'hybrid_paired', 'ft_paired': 'ft_paired'}
    return {float(k): mapping[v] for k, v in decision.items()}


def evaluate(device=None, *, loaded=None, ft_dir=None, train_data=None, data=None, out_root=None, batches=None,
             decision=None):
    import torch
    device = device or rt.base().configure_cuda()
    root = Path(out_root or OUTPUT_ROOT/'reports')
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('A sealed-test report already exists; the test set is evaluated once.')
    batches = batches or TEST_BATCHES
    decision = decision or switch_decision()
    if sorted(decision) != [float(g) for g in cfg.GAMMAS]:
        raise ValueError('Switch decision must cover the six nodes')
    loaded = loaded or trainer.load_terminal_ema(device)
    table, manifest = ext.variants(device, loaded=loaded)
    ft_system, ft_manifest = ft.load_terminal(device, loaded=loaded, directory=ft_dir)
    systems = dict(fdma=table['fdma_baseline'][0], final=table['hybrid_final'][0], ft=ft_system)
    train_data = train_data or rt.make_data(device)
    order, power = bl.rank_coefficients(train_data, device)          # codec_train frames only
    keep = order[:2048]
    analog_f = bl.AnalogCodec(order, power, 24)
    analog_h = bl.AnalogCodec(order, power, 44)
    data = data or test_data(device)
    rows, started = [], time.perf_counter()
    for b in range(batches):
        sample = data.sample('test', b)
        for gi, gamma in enumerate(cfg.GAMMAS):
            index = b*len(cfg.GAMMAS)+gi
            noise, sounding, polarity = streams(index, device)
            g = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
            rgen = torch.Generator().manual_seed(seed(index, 'random_pairing'))
            sgen = torch.Generator().manual_seed(seed(index, 'sscc'))
            for name in tuple(NETWORK)+OTHER:
                with torch.inference_mode():
                    if name == 'perfect_csi_rzf':
                        m = ext.upper_bound_metrics(sample['H_dl'])
                    elif name == 'analog_fdma':
                        m = rt.sample_metrics(bl.analog_forward(analog_f, systems['fdma'], sample, g, noise, polarity))
                    elif name == 'analog_hybrid_paired':
                        m = rt.sample_metrics(bl.analog_forward(analog_h, systems['final'], sample, g, noise, polarity,
                            pairing='least', sounding=sounding))
                    elif name == 'sscc_fdma':
                        out = sc.sscc_forward(systems['fdma'], sample, g, noise, polarity, power[keep], keep, sgen)
                        m = sc.metrics_of(out['H_hat'], out['H_dl'], out['bits'])
                    else:
                        arm, options = NETWORK[name]
                        options = dict(options)
                        if options.pop('sounding', False):
                            options['sounding'] = sounding
                        if options.get('pairing') == 'random':
                            options['random_gen'] = rgen
                        out = ext.forward(systems[arm], sample, g, noise, polarity, **options)
                        m = dict(rt.sample_metrics(out), **ext.extra_rates(out['H_hat'], out['H_dl']))
                for j in range(cfg.MICRO_BATCH):
                    row = dict(variant=name, role='test', gamma_db=float(gamma), sample_index=b*cfg.MICRO_BATCH+j,
                        batch_index=b, **{k: float(v[j]) for k, v in m.items()})
                    if 'fdma' in name:
                        row.pop('key_energy_fraction', None); row.pop('pair_coupling', None)
                    rows.append(row)
        if (b+1) % 16 == 0:
            print(f'test {b+1}/{batches}', flush=True)
    rows += [dict(r, variant='proposed') for r in rows if decision[r['gamma_db']] == r['variant']]
    path = root/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', rows, compress=True)
    metrics = METRICS+('feedback_bits',)
    base_eval.write_csv(path/'summary.csv', results.summarize(rows, 'test', cfg.GAMMAS, metrics))
    base_eval.write_csv(path/'comparisons.csv', results.compare(rows, 'test', cfg.GAMMAS, base_eval.METRICS, CONTRASTS))
    provenance = dict(namespace=NAMESPACE, role='test', batches=batches, mus=batches*cfg.MICRO_BATCH,
        switch_decision={str(k): v for k, v in decision.items()}, switch_source=str(SWITCH_SOURCE),
        final_manifest=manifest, finetune_manifest=ft_manifest, code=rt.source_identity(),
        elapsed_seconds=time.perf_counter()-started, training_updates=0, FDMA_updates=0, test_used=True,
        selection_on_test=False, contains_model_weights=False, contains_channel_tensors=False,
        torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(cfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text('One-time sealed test evaluation of the K8 paper models. All models and the '
        'switch rule were fixed beforehand on validation/selection data.\n')
    files = {p.name: rt.sha256(p) for p in sorted(path.iterdir())}
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=NAMESPACE, export_id=path.name[8:],
        files=files), indent=2, sort_keys=True)+'\n')
    print('REPORT', path.resolve(), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--evaluate-once', action='store_true', required=True)
    parser.parse_args()
    evaluate()


if __name__ == '__main__':
    main()
