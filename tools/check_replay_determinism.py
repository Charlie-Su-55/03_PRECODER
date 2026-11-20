#!/usr/bin/env python3
"""Read-only cross-machine check: does this GPU reproduce the K8 replay frames and noise?

Generates the first four K8 *validation* frames (the 'final' evaluation frames) and the
matching feedback-noise/polarity streams exactly as the evaluations do, and prints content
hashes. The test role is never sampled: for test only the protocol data_id (a hash of the
configuration) is printed. Nothing is written to disk; no model is loaded.

Run the same command on two machines and compare the printed lines.
  exact_sha  : bit-identical tensors
  q4_sha     : identical after rounding to 1e-4 (fragile: ~1e6 values, a 1e-7 difference
               already flips some roundings, so a q4 mismatch alone proves little)
  stats      : energy, max |H|, three fixed random projections and 12 scattered entries,
               printed to many digits; these agree to ~1e-6 relative when only the last
               float bits differ, and differ in the leading digits if the channels differ.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import twc_v1_k8_final_config as cfg  # noqa: E402
import final_runtime as rt  # noqa: E402
import evaluate_twc_v1_k8_test as te  # noqa: E402

FRAMES = 4


def q4(*tensors):
    import torch
    digest = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu()
        if t.is_complex():
            t = torch.view_as_real(t)
        digest.update(torch.round(t.double()*1e4).to(torch.int64).numpy().tobytes())
    return digest.hexdigest()[:16]


def exact(*tensors):
    digest = hashlib.sha256()
    for t in tensors:
        digest.update(t.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def stats(H_dl, H_ul):
    import torch
    x = torch.cat([torch.view_as_real(H_dl).flatten(), torch.view_as_real(H_ul).flatten()]).cpu().double()
    g = torch.Generator().manual_seed(20261120)
    proj = [float(torch.dot(x, torch.randn(x.numel(), generator=g, dtype=torch.float64))) for _ in range(3)]
    idx = torch.linspace(0, x.numel()-1, 12).long()
    return dict(energy=f'{float((x*x).sum()):.9e}', max_abs=f'{float(x.abs().max()):.7f}',
                proj=[f'{v:.7e}' for v in proj], scattered=[f'{float(v):.6f}' for v in x[idx]])


def main():
    device = rt.base().configure_cuda()
    import torch
    import sionna
    props = torch.cuda.get_device_properties(device)
    print(json.dumps(dict(gpu=props.name, sm=props.multi_processor_count, torch=torch.__version__,
                          cuda=torch.version.cuda, sionna=getattr(sionna, '__version__', '?'))))
    data = rt.make_data(device)
    print(json.dumps(dict(validation_data_id=data.protocol['data_id'][:16],
                          test_data_id=te.test_data(device).protocol['data_id'][:16])))
    for b in range(FRAMES):
        s = data.sample('validation', cfg.FINAL_FIRST+b)
        noise, polarity = rt.batch_random('final', b*len(cfg.GAMMAS), device)
        h = s['H_dl'][0, 0, 0, :2].cpu()
        print(json.dumps(dict(frame=b, channel_exact=exact(s['H_dl'], s['H_ul']),
                              channel_q4=q4(s['H_dl'], s['H_ul']),
                              noise_exact=exact(noise, polarity), noise_q4=q4(noise, polarity),
                              H_dl_first=[round(float(v), 6) for v in torch.view_as_real(h).flatten()])))
        print(json.dumps(dict(frame=b, **stats(s['H_dl'], s['H_ul']))))
    print('DONE read-only, nothing written')


if __name__ == '__main__':
    main()
