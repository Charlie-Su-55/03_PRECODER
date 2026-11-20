#!/usr/bin/env python3
"""ONE-TIME sealed test evaluation of the models trained on the H100 (forward passes only).

    python -B evaluate_twc_v1_h100_test.py --evaluate-once k8      # M=4, M=8, joint, DETR
    python -B evaluate_twc_v1_h100_test.py --evaluate-once k16     # K=16 FDMA and Hybrid

K8 arms (hybrid_g2 = M=4, hybrid_g1 = M=8, g4_joint, g4_detr) are scored on exactly the
test frames, feedback noise and pilot polarity of the K8 sealed test run on the new machine
(evaluate_twc_v1_k8_test.test_data/streams). tools/check_replay_determinism.py showed that
the two GPUs reproduce the noise bit-exactly and the channels up to float32 rounding. Each
arm gets its own report with per-batch means (batch_means.csv), so paired differences to the
new-machine FDMA can be formed batch by batch.

K16: FDMA and Hybrid of the K=16 study on the test role of the K=16 data (two K8 test draws
per MU, own test-only noise namespace). The FDMA/Hybrid switch is read from the K=16
validation report; nothing is selected on test.

Every report directory is evaluated once: an existing report stops the run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import uuid

from configs import twc_v1_k8_final_config as kcfg
from configs import twc_v1_k8_grouped_config as gcfg
import final_runtime as fr
import evaluate_twc_v1_k8_test as te

ROOT = Path(__file__).resolve().parent
K8_NAMESPACE = 'twc_v1_k8_test_h100_v1'
K16_NAMESPACE = 'twc_v1_k16_test_v1'
K8_ARMS = ('hybrid_g2', 'hybrid_g1', 'g4_joint', 'g4_detr')
K16_SWITCH_SOURCE = ROOT/'runs/twc_v1_k16_v1/reports/attempt_63ef716343a8/provenance.json'
TEST_BATCHES = 256
BATCH_METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'min_user_rate_bps_hz')


def batch_means(rows, metrics=BATCH_METRICS, batch_size=4):
    """Per (variant, gamma, batch) means; paired batch CIs can be rebuilt from these exactly."""
    acc = {}
    for r in rows:
        for m in metrics:
            if m in r:
                acc.setdefault((r['variant'], r['role'], float(r['gamma_db']), int(r['batch_index']), m), []).append(
                    float(r[m]))
    out = []
    for (v, role, g, b, m), x in sorted(acc.items()):
        if len(x) != batch_size:
            raise ValueError('Incomplete batch in batch means')
        out.append(dict(variant=v, role=role, gamma_db=g, batch_index=b, metric=m, n_samples=len(x),
                        mean=statistics.mean(x)))
    return out


def _fresh(root):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'A sealed-test report already exists in {root}; the test set is evaluated once.')
    return root


def _finish(path, namespace, provenance, readme):
    (path/'provenance.json').write_text(json.dumps(kcfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text(readme+'\n')
    files = {p.name: fr.sha256(p) for p in sorted(path.iterdir())}
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=namespace, export_id=path.name[8:],
        files=files), indent=2, sort_keys=True)+'\n')
    print('REPORT', path.resolve(), flush=True)
    return path


# ------------------------------------------------------------------ K8 arms --

def use_study(arm):
    """Point the shared grouped trainer/runtime at the study that owns `arm` (this process only)."""
    import grouped_runtime as gr
    import train_twc_v1_k8_grouped as gt
    import train_twc_v1_k8_decoder_cmp as dc
    if arm in gcfg.ARMS:
        gt.cfg, gr.cfg, gt.build = gcfg, gcfg, _GROUPED_BUILD
    elif arm in dc.cfg.ARMS:
        dc.install()
    else:
        raise ValueError('Unknown H100 K8 arm '+arm)
    return gt, gr


def _grouped_build():
    import train_twc_v1_k8_grouped as gt
    return gt.build


_GROUPED_BUILD = _grouped_build()


def evaluate_k8_arm(arm, device=None, *, loaded=None, data=None, out_root=None, batches=None):
    import torch
    import evaluate_twc_v1_k8_final as base_eval
    import twc_v1_k8_final_results as results
    gt, gr = use_study(arm)
    device = device or fr.base().configure_cuda()
    root = _fresh(Path(out_root or ROOT/'runs'/K8_NAMESPACE/'reports')/arm)
    batches = batches or TEST_BATCHES
    system, manifest = loaded or gt.load_terminal(arm, device)
    data = data or te.test_data(device)
    rows, started = [], time.perf_counter()
    for b in range(batches):
        sample = data.sample('test', b)
        for gi, gamma in enumerate(kcfg.GAMMAS):
            noise, _, polarity = te.streams(b*len(kcfg.GAMMAS)+gi, device)
            g = torch.full((kcfg.MICRO_BATCH,), float(gamma), device=device)
            with torch.inference_mode():
                m = fr.sample_metrics(gr.forward(system, sample, g, noise, polarity))
            for j in range(kcfg.MICRO_BATCH):
                row = dict(variant=arm, role='test', gamma_db=float(gamma), sample_index=b*kcfg.MICRO_BATCH+j,
                           batch_index=b, **{k: float(v[j]) for k, v in m.items()})
                row.pop('pair_coupling', None)
                rows.append(row)
        if (b+1) % 32 == 0:
            print(f'{arm} test {b+1}/{batches}', flush=True)
    path = root/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', rows, compress=True)
    base_eval.write_csv(path/'summary.csv', results.summarize(rows, 'test', kcfg.GAMMAS, base_eval.METRICS))
    base_eval.write_csv(path/'batch_means.csv', batch_means(rows))
    provenance = dict(namespace=K8_NAMESPACE, arm=arm, role='test', batches=batches, mus=batches*kcfg.MICRO_BATCH,
        frames='K8 sealed test frames, noise and polarity of evaluate_twc_v1_k8_test (same seeds)',
        test_data_id=data.protocol['data_id'] if hasattr(data, 'protocol') else None,
        terminal=manifest, code=fr.source_identity(), elapsed_seconds=time.perf_counter()-started,
        training_updates=0, FDMA_updates=0, test_used=True, selection_on_test=False,
        contains_model_weights=False, contains_channel_tensors=False, torch=torch.__version__)
    return _finish(path, K8_NAMESPACE, provenance, f'One-time sealed test of the H100 K8 arm {arm}; '
                   'trained and fixed beforehand on training/validation data.')


# --------------------------------------------------------------------- K16 --

class K16TestData:
    """Test role of the K=16 data: MU i is the concatenation of K8 test draws 2i and 2i+1."""
    def __init__(self, device):
        from configs import twc_v1_k16_config as c16
        source = fr.base().original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
        B = c16.MICRO_BATCH
        self._base = source.TWCv1K8ReplayableData(base_seed=c16.SEED,
            batch_sizes=dict(ce_train=B, codec_train=B, ce_calibration=B, validation=B, test=B),
            device=str(device), cdl_model='C', speed=1., seed_namespace=c16.NAMESPACE)
        self.protocol = self._base.protocol

    def sample(self, role, index):
        import torch
        if role != 'test':
            raise ValueError('K16TestData serves the test role only')
        a, b = self._base.sample('test', 2*index), self._base.sample('test', 2*index+1)
        return dict(H_dl=torch.cat((a['H_dl'], b['H_dl']), 1), H_ul=torch.cat((a['H_ul'], b['H_ul']), 1))


def k16_switch(path=K16_SWITCH_SOURCE):
    decision = json.loads(Path(path).read_text())['switch_decision']
    out = {float(k): v for k, v in decision.items()}
    if any(v not in ('fdma_baseline', 'hybrid_paired') for v in out.values()):
        raise ValueError('Unexpected K16 switch entry')
    return out


def evaluate_k16(device=None, *, systems=None, data=None, out_root=None, batches=None, decision=None):
    import torch
    import evaluate_twc_v1_k8_final as base_eval
    import twc_v1_k8_final_results as results
    import k16_runtime as r
    import train_twc_v1_k16 as t16
    from configs import twc_v1_k16_config as c16
    device = device or fr.base().configure_cuda()
    root = _fresh(out_root or ROOT/'runs'/K16_NAMESPACE/'reports')
    batches = batches or TEST_BATCHES
    decision = decision or k16_switch()
    if sorted(decision) != [float(g) for g in c16.GAMMAS]:
        raise ValueError('K16 switch decision must cover the six nodes')
    receipts = {}
    if systems is None:
        bundle = t16.load_ce(device)
        systems = {}
        for arm in c16.ARMS:
            systems[arm], receipts[arm] = t16.load_codec(arm, device, bundle)
    data = data or K16TestData(device)
    rows, started = [], time.perf_counter()
    for b in range(batches):
        sample = data.sample('test', b)
        for gi, gamma in enumerate(c16.GAMMAS):
            noise, sounding, polarity = r.random_streams(K16_NAMESPACE, c16.SEED, 'test', b*len(c16.GAMMAS)+gi,
                                                         c16.MICRO_BATCH, device)
            g = torch.full((c16.MICRO_BATCH,), float(gamma), device=device)
            for name, (arm, options) in t16.VARIANTS.items():
                with torch.inference_mode():
                    if arm is None:
                        m = r.upper_bound_metrics(sample['H_dl'])
                    else:
                        m = r.sample_metrics(r.forward(systems[arm], sample, g, noise, polarity,
                                                       sounding=sounding, **options))
                for j in range(c16.MICRO_BATCH):
                    row = dict(variant=name, role='test', gamma_db=float(gamma), sample_index=b*c16.MICRO_BATCH+j,
                               batch_index=b, **{k: float(v[j]) for k, v in m.items()})
                    if name.startswith('fdma'):
                        row.pop('key_energy_fraction', None); row.pop('pair_coupling', None)
                    rows.append(row)
        if (b+1) % 16 == 0:
            print(f'k16 test {b+1}/{batches}', flush=True)
    rows += [dict(x, variant='adaptive') for x in rows if decision[x['gamma_db']] == x['variant']]
    path = root/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', rows, compress=True)
    base_eval.write_csv(path/'summary.csv', results.summarize(rows, 'test', c16.GAMMAS, t16.METRICS))
    base_eval.write_csv(path/'comparisons.csv', results.compare(rows, 'test', c16.GAMMAS, t16.METRICS, t16.CONTRASTS))
    provenance = dict(namespace=K16_NAMESPACE, K=r.K, role='test', batches=batches, mus=batches*c16.MICRO_BATCH,
        switch_decision={str(k): v for k, v in decision.items()}, switch_source=str(K16_SWITCH_SOURCE),
        test_data_id=data.protocol['data_id'] if hasattr(data, 'protocol') else None, receipts=receipts,
        elapsed_seconds=time.perf_counter()-started, training_updates=0, test_used=True, selection_on_test=False,
        contains_model_weights=False, contains_channel_tensors=False, torch=torch.__version__)
    return _finish(path, K16_NAMESPACE, provenance, 'One-time sealed test of the K=16 user-load study; '
                   'models and switch fixed beforehand on training/validation data.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--evaluate-once', choices=('k8', 'k16'), required=True)
    args = parser.parse_args(argv)
    if args.evaluate_once == 'k8':
        roots = [ROOT/'runs'/K8_NAMESPACE/'reports'/arm for arm in K8_ARMS]
        for root in roots:            # refuse before any work if any arm was already tested
            _fresh(root)
        for arm in K8_ARMS:
            evaluate_k8_arm(arm)
    else:
        evaluate_k16()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
