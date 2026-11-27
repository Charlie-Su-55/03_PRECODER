#!/usr/bin/env python3
"""Key-count ablation on the paired Hybrid (q = 0, 2, 8 key REs per UE), from scratch.

    python -B train_twc_v1_k8_key_ablation.py --arm q0 --train [--resume]
    python -B train_twc_v1_k8_key_ablation.py --arm q0 --evaluate             # validation frames
    python -B train_twc_v1_k8_key_ablation.py --arm q0 --evaluate-once-test   # sealed test, once

Reuses the group-size trainer and evaluator unchanged; this process swaps in the
key-ablation configuration, the q-general layout/waveform/receiver (keyed_runtime)
and g4_joint's training data stream. The q = 4 reference is g4_joint of the
decoder comparison (same recipe, initialization and training channels). The
validation report uses the final model's validation frames and noise, the test
report the K8 sealed test frames, noise and polarity (same as every other K8 test
report), so all arms and FDMA can be paired batch by batch. FDMA is not trained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

from configs import twc_v1_k8_key_ablation_config as cfg
import final_runtime as fr
import grouped_runtime as gr
import keyed_runtime as kr
import train_twc_v1_k8_grouped as gt

TEST_BATCHES = 256


def build(arm, device, ce, table):
    import torch
    from models.twc_v1_k8_icc_aligned import create_codec
    layout = kr.KeyLayout(cfg.ARMS[arm], cfg.GROUP_COUNT)
    codec = create_codec('hybrid_g4', device=str(device), seed=cfg.CODEC_INIT_SEED)
    kr.set_geometry(codec.decoder, layout)
    model = fr.FinalCodec(codec).to(device)
    n = kr.active_dimension(layout)
    stats = kr.NodeStatistics(layout, table.detach().cpu(), torch.full((8, n), 48./n)).to(device)
    return dict(kind='grouped', codec=model, ce=ce, stats=stats, layout=layout)


def make_train_data(device):
    """g4_joint's training channel stream (decoder-comparison seed and namespace)."""
    source = fr.base().original.load_local('utils/twc_v1_k8_replayable_data.py', '_icc_aligned_data')
    return source.TWCv1K8ReplayableData(base_seed=cfg.DATA_SEED,
        batch_sizes=dict(ce_train=4, codec_train=cfg.MICRO_BATCH, ce_calibration=4, validation=4, test=1),
        device=str(device), cdl_model='C', speed=1., seed_namespace=cfg.DATA_NAMESPACE)


_SAVED = {}


def install():
    """Point the group-size trainer/runtime at this study (this process only); returns a restore function."""
    if not _SAVED:
        _SAVED.update(gt_cfg=gt.cfg, gr_cfg=gr.cfg, build=gt.build, forward=gr.forward,
                      active_dimension=gr.active_dimension, make_train_data=gr.make_train_data)
    gt.cfg, gr.cfg, gt.build = cfg, cfg, build
    gr.forward, gr.active_dimension, gr.make_train_data = kr.forward, kr.active_dimension, make_train_data

    def restore():
        gt.cfg, gr.cfg, gt.build = _SAVED['gt_cfg'], _SAVED['gr_cfg'], _SAVED['build']
        gr.forward, gr.active_dimension = _SAVED['forward'], _SAVED['active_dimension']
        gr.make_train_data = _SAVED['make_train_data']
    return restore


def evaluate_once_test(arm, device=None, *, loaded=None, data=None, out_root=None, batches=None):
    """One-time sealed test on the K8 test frames, noise and polarity (as the H100 test report)."""
    import torch
    import evaluate_twc_v1_k8_final as base_eval
    import evaluate_twc_v1_k8_test as te
    import evaluate_twc_v1_h100_test as ht
    import twc_v1_k8_final_results as results
    from configs import twc_v1_k8_final_config as kcfg
    install()
    device = device or fr.base().configure_cuda()
    root = ht._fresh(Path(out_root or cfg.ROOT/'runs'/cfg.TEST_NAMESPACE/'reports')/arm)
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
                m = fr.sample_metrics(kr.forward(system, sample, g, noise, polarity))
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
    base_eval.write_csv(path/'batch_means.csv', ht.batch_means(rows, ht.BATCH_METRICS+('direction_error',)))
    provenance = dict(namespace=cfg.TEST_NAMESPACE, arm=arm, layout=system['layout'].to_dict(), role='test',
        batches=batches, mus=batches*kcfg.MICRO_BATCH,
        frames='K8 sealed test frames, noise and polarity of evaluate_twc_v1_k8_test (same seeds)',
        test_data_id=data.protocol['data_id'] if hasattr(data, 'protocol') else None,
        terminal=manifest, recipe=cfg.recipe(arm), code=fr.source_identity(),
        elapsed_seconds=time.perf_counter()-started, training_updates=0, FDMA_updates=0, test_used=True,
        selection_on_test=False, contains_model_weights=False, contains_channel_tensors=False,
        torch=torch.__version__)
    return ht._finish(path, cfg.TEST_NAMESPACE, provenance, f'One-time sealed test of the key-ablation arm {arm} '
                      f'(q={cfg.ARMS[arm]} key REs per UE, pairs); trained and fixed beforehand on training data.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', choices=tuple(cfg.ARMS), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store_true')
    action.add_argument('--evaluate', action='store_true')
    action.add_argument('--evaluate-once-test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args(argv)
    install()
    if args.train:
        print(json.dumps(gt.train(args.arm, args.stop_after, args.resume), indent=2, sort_keys=True))
    elif args.evaluate:
        gt.evaluate(args.arm)
    else:
        evaluate_once_test(args.arm)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
