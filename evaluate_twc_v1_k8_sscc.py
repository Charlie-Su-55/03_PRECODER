#!/usr/bin/env python3
"""Idealized separate source-channel coding (SSCC) baseline on the FDMA frame (forward passes only).

An optimistic bound for any digital feedback (codebook, CsiNet with quantization and channel coding):
  channel coding  each UE's 24 FDMA REs carry R_k = sum_n log2(1 + e*||h_hat_kn||^2/(N0 + e*sigma2_kn))
                  bits, e = E_d/24, with the same pilot-only UL estimate and calibrated CE error as
                  the analog schemes (capacity-achieving code, no outage, infinite blocklength);
  source coding   rate-distortion-optimal Gaussian coding of the angular-delay coefficients with
                  ensemble variances ranked on codec_train frames (reverse water-filling) and the
                  Gaussian backward test channel; no learned decoder.
Same frames and noise as the final, extended and analog reports. Test sealed; FDMA weights untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import uuid

from configs import twc_v1_k8_final_config as cfg
import final_runtime as rt
import evaluate_twc_v1_k8_final as base_eval
import evaluate_twc_v1_k8_extended as ext
import evaluate_twc_v1_k8_baselines as bl
import twc_v1_k8_final_results as results

NAMESPACE = 'twc_v1_k8_sscc_v1'
OUTPUT_ROOT = Path('runs')/NAMESPACE
VARIANTS = ('fdma_baseline', 'hybrid_paired', 'sscc_fdma', 'analog_fdma', 'perfect_csi_rzf')
CONTRASTS = (('fdma_minus_sscc', 'fdma_baseline', 'sscc_fdma'),
             ('paired_minus_sscc', 'hybrid_paired', 'sscc_fdma'),
             ('sscc_minus_analog', 'sscc_fdma', 'analog_fdma'))
METRICS = ('sum_rate_bps_hz', 'dl_nmse_linear', 'direction_error', 'interference_power_mean',
           'min_user_rate_bps_hz', 'feedback_bits')


def water_fill(lam, bits):
    """Distortions d [B,K,N] with sum log2(lam/d) = bits for complex Gaussian components."""
    import torch
    log_lam = torch.log2(lam.clamp_min(1e-30))
    lo = torch.full(bits.shape, -60., dtype=torch.float64, device=bits.device)
    hi = torch.full(bits.shape, float(log_lam.max())+1, dtype=torch.float64, device=bits.device)
    for _ in range(80):                         # bisection on log2(theta)
        mid = (lo+hi)/2
        rate = (log_lam[None, None]-mid[..., None]).clamp_min(0).sum(-1)
        lo, hi = torch.where(rate > bits, mid, lo), torch.where(rate > bits, hi, mid)
    theta = torch.pow(2., hi)
    return torch.minimum(lam[None, None].expand(*bits.shape, -1), theta[..., None])


def sscc_forward(system, sample, gamma, noise, polarity, lam, order, gen):
    import torch
    phy = rt.physics()
    ce, stats = system['ce'], system['stats']
    H_dl, H_ul = sample['H_dl'], sample['H_ul']
    batch = H_ul.shape[0]
    pilots = phy.known_pilots(batch, H_ul.device, polarity)
    Y = phy.transmit(H_ul, pilots, noise, gamma)
    est = ce(Y, pilots, phy.fourier_ls(Y, pilots), gamma_db=gamma)
    cevar = stats.ce_variance(gamma)                                  # [B,K,Nsc]
    layout = stats.layout
    index = torch.tensor(layout.data_positions, device=H_ul.device)     # [K,24]
    e = rt.ED/index.shape[-1]
    n0 = .25*torch.pow(torch.tensor(10., device=H_ul.device), -gamma/10)
    gain = est.abs().square().sum(2)                                   # [B,K,Nsc]
    g = gain.gather(-1, index[None].expand(batch, -1, -1))
    v = cevar.gather(-1, index[None].expand(batch, -1, -1))
    bits = torch.log2(1+e*g/(n0[:, None, None]+e*v)).sum(-1).double()  # [B,K]
    d = water_fill(lam.double(), bits)                                 # [B,K,N]
    C = bl.to_coefficients(H_dl).reshape(batch, rt.K, -1)[..., order]
    a = (1-d/lam[None, None].double()).clamp_min(0).float()
    w = torch.randn(C.shape+(2,), generator=gen, device='cpu').to(C.device)
    w = torch.view_as_complex(w.contiguous())/2**.5
    hat = a*C+(a*d.float()).clamp_min(0).sqrt()*w
    out = torch.zeros((batch, rt.K, rt.NT*rt.NSC), dtype=C.dtype, device=C.device)
    out[..., order] = hat
    H_hat = bl.from_coefficients(out.reshape(H_dl.shape))
    return dict(H_hat=H_hat, H_dl=H_dl, bits=bits.float())


def metrics_of(H_hat, H_dl, bits=None):
    import torch
    from twc_v1_k8_hybrid_direction import direction_error
    _, r = rt.base().helpers().rate_objective(H_hat, H_dl, alpha=cfg.ALPHA, downlink_snr_db=cfg.DL_SNR)
    nmse = ((H_hat-H_dl).abs().square().sum((2, 3))/H_dl.abs().square().sum((2, 3))).mean(1)
    m = dict(sum_rate_bps_hz=r['sum_rate_bps_hz'], dl_nmse_linear=nmse,
        direction_error=direction_error(H_hat, H_dl).to(nmse.dtype),
        interference_power_mean=r['interference_power_mean'], min_user_rate_bps_hz=r['user_rate_bps_hz'].min(-1).values)
    if bits is not None:
        m['feedback_bits'] = bits.mean(-1)
    return {k: v.detach().float().cpu() for k, v in m.items()}


def run(table, analog, lam, order, data, role, first, batches, gammas, device):
    import torch
    rows = []
    for b in range(batches):
        sample = data.sample('validation', first+b)
        for gi, gamma in enumerate(gammas):
            index = b*len(gammas)+gi
            noise, polarity = rt.batch_random(role, index, device)
            sound = ext.sounding_noise(role, index, device)
            g = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
            gen = torch.Generator().manual_seed(ext._seed(role, index, 'sscc_test_channel'))
            for name in VARIANTS:
                with torch.inference_mode():
                    if name == 'perfect_csi_rzf':
                        m = metrics_of(sample['H_dl'], sample['H_dl'])
                    elif name == 'sscc_fdma':
                        out = sscc_forward(table['fdma_baseline'][0], sample, g, noise, polarity, lam, order, gen)
                        m = metrics_of(out['H_hat'], out['H_dl'], out['bits'])
                    elif name == 'analog_fdma':
                        codec, system, _ = analog
                        out = bl.analog_forward(codec, system, sample, g, noise, polarity)
                        m = metrics_of(out['H_hat'], out['H_dl'])
                    else:
                        system, options = table[name]
                        options = dict(options)
                        if options.pop('sounding', False):
                            options['sounding'] = sound
                        out = ext.forward(system, sample, g, noise, polarity, **options)
                        m = metrics_of(out['H_hat'], out['H_dl'])
                for j in range(cfg.MICRO_BATCH):
                    rows.append(dict(variant=name, role=role, gamma_db=float(gamma), sample_index=b*cfg.MICRO_BATCH+j,
                        batch_index=b, **{k: float(v[j]) for k, v in m.items()}))
        if (b+1) % 16 == 0:
            print(f'{role} {b+1}/{batches}', flush=True)
    return rows


def evaluate(device=None, *, loaded=None, data=None, out_root=None, sizes=None, rank_batches=None):
    import torch
    device = device or rt.base().configure_cuda()
    sizes = sizes or dict(final=cfg.FINAL_BATCHES)
    table, manifest = ext.variants(device, loaded=loaded)
    data = data or rt.make_data(device)
    order_all, power = bl.rank_coefficients(data, device, **({} if rank_batches is None else dict(batches=rank_batches)))
    keep = order_all[:2048]                       # components beyond this carry negligible ensemble power
    lam = power[keep]                             # absolute ensemble variance per kept coefficient
    analog = (bl.AnalogCodec(order_all, power, 24), table['fdma_baseline'][0], {})
    started = time.perf_counter()
    rows = run(table, analog, lam, keep, data, 'final', cfg.FINAL_FIRST, sizes['final'], cfg.GAMMAS, device)
    path = Path(out_root or OUTPUT_ROOT/'reports')/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', rows, compress=True)
    base_eval.write_csv(path/'summary.csv', results.summarize(rows, 'final', cfg.GAMMAS, METRICS))
    base_eval.write_csv(path/'comparisons.csv', results.compare(rows, 'final', cfg.GAMMAS, METRICS[:5], CONTRASTS))
    provenance = dict(namespace=NAMESPACE, variants=list(VARIANTS), kept_components=int(len(keep)),
        kept_energy=float(power[keep].sum()/power.sum()), final_manifest=manifest, code=rt.source_identity(),
        elapsed_seconds=time.perf_counter()-started, training_updates=0, FDMA_updates=0, test_used=False,
        idealization='capacity-achieving channel code and RD-optimal Gaussian quantizer (optimistic for SSCC)',
        contains_model_weights=False, contains_channel_tensors=False, torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(cfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text('Idealized SSCC baseline on the FDMA frame (validation role; test sealed).\n')
    files = {p.name: rt.sha256(p) for p in sorted(path.iterdir())}
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=NAMESPACE, export_id=path.name[8:],
        files=files), indent=2, sort_keys=True)+'\n')
    print('REPORT', path.resolve(), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--evaluate', action='store_true', required=True)
    parser.parse_args()
    evaluate()


if __name__ == '__main__':
    main()
