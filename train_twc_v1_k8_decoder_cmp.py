#!/usr/bin/env python3
"""Decoder comparison (joint vs DETR-style) on the paired Hybrid layout, both from scratch.

    python -B train_twc_v1_k8_decoder_cmp.py --arm g4_detr --train [--resume]
    python -B train_twc_v1_k8_decoder_cmp.py --arm g4_detr --evaluate

Reuses the group-size trainer unchanged; only the configuration module and the
decoder construction are swapped in this process. FDMA is not trained.
"""
from __future__ import annotations

import sys

from configs import twc_v1_k8_decoder_cmp_config as cfg
import final_runtime as fr
import grouped_runtime as gr
import train_twc_v1_k8_grouped as gt


def build(arm, device, ce, table):
    import torch
    from models.twc_v1_k8_icc_aligned import create_codec
    from models.twc_v1_k8_detr_decoder import K8DETRDecoder
    layout = gr.make_layout('hybrid_g4')
    codec = create_codec('hybrid_g4', device=str(device), seed=cfg.CODEC_INIT_SEED)
    if cfg.DECODERS[arm] == 'detr':
        torch.manual_seed(cfg.CODEC_INIT_SEED+1)
        codec.decoder = K8DETRDecoder(layout).to(device)
    model = fr.FinalCodec(codec).to(device)
    n = gr.active_dimension(layout)
    stats = gr.NodeStatistics(layout, table.detach().cpu(), torch.full((8, n), 48./n)).to(device)
    return dict(kind='grouped', codec=model, ce=ce, stats=stats, layout=layout)


def install():
    """Point the group-size trainer and runtime at this study's configuration (this process only)."""
    gt.cfg = cfg
    gr.cfg = cfg
    gt.build = build


def main(argv=None):
    install()
    return gt.main(argv)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
