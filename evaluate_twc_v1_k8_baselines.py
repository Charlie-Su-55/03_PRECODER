#!/usr/bin/env python3
"""Training-free analog feedback baselines on the same frame (forward passes only).

Classical linear analog feedback in the spirit of Kumar et al. and
Khalilsarai et al.: the UE sends the d strongest angular-delay coefficients of
its DL channel (fixed index set, ranked by mean power on codec_train frames,
never on validation or test) as uncoded complex symbols on its d data REs,
normalized to E_d per realization exactly as the learned encoders. The BS
uses the same pilot-only estimator, the same error-aware LMMSE receiver with
a prior proportional to the coefficient powers, rescales by the expected
energy factor, places the coefficients back and inverts the transform. No
learned decoder. Precoding is the same RZF.

Variants (same frames/noise as the final and extended reports):
  analog_fdma            24 coefficients on the FDMA layout
  analog_hybrid          44 coefficients on the Hybrid layout (4 key + 40 shared), fixed pairs
  analog_hybrid_paired   as above with least-coupled pairing (independent sounding noise)
  fdma_baseline, hybrid_paired, perfect_csi_rzf   learned references
Test sealed; FDMA weights untouched.
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
import twc_v1_k8_final_results as results

NAMESPACE = 'twc_v1_k8_analog_baselines_v1'
OUTPUT_ROOT = Path('runs')/NAMESPACE
RANK_BATCHES = 64
VARIANTS = ('fdma_baseline', 'hybrid_paired', 'analog_fdma', 'analog_hybrid', 'analog_hybrid_paired', 'perfect_csi_rzf')
CONTRASTS = (('fdma_minus_analog_fdma', 'fdma_baseline', 'analog_fdma'),
             ('paired_minus_analog_paired', 'hybrid_paired', 'analog_hybrid_paired'),
             ('analog_paired_minus_analog_fdma', 'analog_hybrid_paired', 'analog_fdma'),
             ('analog_paired_minus_analog_fixed', 'analog_hybrid_paired', 'analog_hybrid'),
             ('paired_minus_analog_fdma', 'hybrid_paired', 'analog_fdma'))
METRICS = base_eval.METRICS


def to_coefficients(H):
    import torch
    return torch.fft.ifft(torch.fft.fft(H, dim=-2), dim=-1)          # [B,K,Nt,Nsc] angular x delay


def from_coefficients(C):
    import torch
    return torch.fft.ifft(torch.fft.fft(C, dim=-1), dim=-2)


def rank_coefficients(data, device, batches=RANK_BATCHES):
    """Mean power of every angular-delay coefficient over codec_train frames (users pooled)."""
    import torch
    power = torch.zeros(rt.NT*rt.NSC, device=device)
    for b in range(batches):
        H = data.sample('codec_train', 1+b)['H_dl']
        power += to_coefficients(H).abs().square().mean((0, 1)).reshape(-1)
    power /= batches
    order = torch.argsort(power, descending=True)
    return order, power


class AnalogCodec:
    def __init__(self, order, power, d):
        self.index = order[:d]
        self.power = power[self.index]
        self.scale = (rt.ED/self.power.sum()).sqrt()               # expected per-realization factor

    def encode(self, H_dl):
        C = to_coefficients(H_dl).reshape(*H_dl.shape[:2], -1)
        raw = C.new_zeros((*C.shape[:2], rt.DMAX))
        raw[..., :len(self.index)] = C[..., self.index]
        return raw

    def prior(self, batch):
        p = self.power/self.power.sum()*rt.ED
        return p[None, None].expand(batch, rt.K, -1).contiguous().float()

    def decode(self, soft, shape):
        d = len(self.index)
        coeff = soft[..., :d]/self.scale
        C = coeff.new_zeros((*shape[:2], rt.NT*rt.NSC))
        C[..., self.index] = coeff
        return from_coefficients(C.reshape(shape))


def analog_forward(codec, layout_system, sample, gamma, noise, polarity, *, pairing=None, sounding=None):
    import torch
    phy = rt.physics()
    ce, stats = layout_system['ce'], layout_system['stats']
    H_dl, H_ul = sample['H_dl'], sample['H_ul']
    if pairing is not None:
        pilots = phy.known_pilots(H_ul.shape[0], H_ul.device, polarity)
        Y0 = phy.transmit(H_ul, pilots, noise if sounding is None else sounding, gamma)
        est0 = ce(Y0, pilots, phy.fourier_ls(Y0, pilots), gamma_db=gamma)
        perm = ext.choose_pairing(est0, pairing)
        H_dl, H_ul = rt.permute_users(H_dl, perm), rt.permute_users(H_ul, perm)
    wave = phy.build_waveform(codec.encode(H_dl), stats.layout, polarity=polarity)
    Y = phy.transmit(H_ul, wave.x_grid, noise, gamma)
    estimated = ce(Y, wave.pilot, phy.fourier_ls(Y, wave.pilot), gamma_db=gamma)
    cevar = stats.ce_variance(gamma)
    recovered = rt.recover_batched(Y, estimated, stats.layout, codec.prior(H_dl.shape[0]), cevar, gamma)
    H_hat = codec.decode(recovered['soft'], H_dl.shape)
    return dict(H_hat=H_hat, H_dl=H_dl, wave=wave, estimated=estimated, gamma=gamma,
        pair_coupling=rt.slot_pair_coupling(estimated))


def run(table, analog, data, role, first, batches, gammas, device):
    import torch
    rows = []
    for b in range(batches):
        sample = data.sample('validation', first+b)
        for gi, gamma in enumerate(gammas):
            index = b*len(gammas)+gi
            noise, polarity = rt.batch_random(role, index, device)
            sound = ext.sounding_noise(role, index, device)
            g = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
            for name in VARIANTS:
                with torch.inference_mode():
                    if name == 'perfect_csi_rzf':
                        metrics = ext.upper_bound_metrics(sample['H_dl'])
                        metrics = {k: v for k, v in metrics.items() if not k.startswith('sum_rate_dl')}
                    elif name.startswith('analog'):
                        codec, system, options = analog[name]
                        options = dict(options)
                        if options.pop('sounding', False):
                            options['sounding'] = sound
                        metrics = rt.sample_metrics(analog_forward(codec, system, sample, g, noise, polarity, **options))
                    else:
                        system, options = table[name]
                        options = dict(options)
                        if options.pop('sounding', False):
                            options['sounding'] = sound
                        metrics = rt.sample_metrics(ext.forward(system, sample, g, noise, polarity, **options))
                for j in range(cfg.MICRO_BATCH):
                    row = dict(variant=name, role=role, gamma_db=float(gamma), sample_index=b*cfg.MICRO_BATCH+j,
                        batch_index=b, **{k: float(v[j]) for k, v in metrics.items()})
                    if 'fdma' in name:
                        row.pop('key_energy_fraction', None); row.pop('pair_coupling', None)
                    rows.append(row)
        if (b+1) % 16 == 0:
            print(f'{role} {b+1}/{batches}', flush=True)
    return rows


def evaluate(device=None, *, loaded=None, data=None, out_root=None, sizes=None):
    import torch
    device = device or rt.base().configure_cuda()
    sizes = sizes or dict(final=cfg.FINAL_BATCHES, midpoint=cfg.MID_BATCHES)
    table, manifest = ext.variants(device, loaded=loaded)
    data = data or rt.make_data(device)
    order, power = rank_coefficients(data, device)
    fdma_system, hybrid_system = table['fdma_baseline'][0], table['hybrid_final'][0]
    analog = dict(analog_fdma=(AnalogCodec(order, power, 24), fdma_system, {}),
        analog_hybrid=(AnalogCodec(order, power, 44), hybrid_system, {}),
        analog_hybrid_paired=(AnalogCodec(order, power, 44), hybrid_system, dict(pairing='least', sounding=True)))
    captured = dict(top24=float(power[order[:24]].sum()/power.sum()), top44=float(power[order[:44]].sum()/power.sum()))
    started = time.perf_counter()
    final = run(table, analog, data, 'final', cfg.FINAL_FIRST, sizes['final'], cfg.GAMMAS, device)
    mid = run(table, analog, data, 'midpoint', cfg.MID_FIRST, sizes['midpoint'], cfg.MID_GAMMAS, device)
    path = Path(out_root or OUTPUT_ROOT/'reports')/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', final+mid, compress=True)
    base_eval.write_csv(path/'summary.csv', results.summarize(final, 'final', cfg.GAMMAS, METRICS)
        + results.summarize(mid, 'midpoint', cfg.MID_GAMMAS, METRICS))
    base_eval.write_csv(path/'comparisons.csv', results.compare(final, 'final', cfg.GAMMAS, METRICS, CONTRASTS)
        + results.compare(mid, 'midpoint', cfg.MID_GAMMAS, METRICS, CONTRASTS))
    provenance = dict(namespace=NAMESPACE, variants=list(VARIANTS), rank_frames=dict(role='codec_train',
        first=1, batches=RANK_BATCHES), captured_mean_energy=captured, final_manifest=manifest,
        code=rt.source_identity(), elapsed_seconds=time.perf_counter()-started, training_updates=0,
        FDMA_updates=0, test_used=False, contains_model_weights=False, contains_channel_tensors=False,
        torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(cfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text('Training-free analog feedback baselines (validation role; test sealed).\n')
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
