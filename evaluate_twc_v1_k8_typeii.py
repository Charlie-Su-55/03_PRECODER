#!/usr/bin/env python3
"""3GPP NR Type II codebook baseline (Rel-15, rank 1) on the FDMA frame (forward passes only).

UE side (38.214 Sec. 5.2.2.2.3, adapted to the single-polarized 4x8 BS array):
  * oversampled 2D DFT beams, O1 = O2 = 4, N1 x N2 from the BS antenna grid (8 x 4);
  * wideband: rotation (q1, q2) and L orthogonal beams that maximize the wideband beam power,
    strongest-beam indicator, 3-bit wideband amplitudes {1, 1/2^(1/2), ..., 1/64^(1/2), 0};
  * per subband of 4 PRBs (48 subcarriers): projection of the principal eigenvector of the
    subband covariance onto the selected beams, phases relative to the strongest beam
    quantized to N_PSK, optional 1-bit subband amplitudes {1, 1/2^(1/2)}.
  With one polarization the codebook has L coefficients instead of 2L; since K(2) >= L for
  all configurations, every non-zero, non-strongest coefficient carries the N_PSK phase and,
  if enabled, the subband amplitude.
Feedback link (optimistic, identical to the SSCC baseline): each UE's 24 FDMA data REs carry
  their capacity with the same pilot-only UL estimate and calibrated estimation error; every UE
  uses the highest-resolution parameter combination (Table 5.2.2.2.3-5) whose payload fits.
BS side: H_hat[:, m] = g_sb * w_sb on the subcarriers of each subband, with the ideal subband
  channel gain g_sb (unquantized CQI); RZF as for all schemes.
Diagnostic variant `subband_eigvec`: the unquantized subband eigenvector with the same gain,
  i.e. ideal rank-1 subband CSI, an upper bound for any subband PMI.

    python -B evaluate_twc_v1_k8_typeii.py --validate              # validation frames, repeatable
    python -B evaluate_twc_v1_k8_typeii.py --evaluate-once-test    # sealed test set, once

No training; FDMA weights untouched; the configuration is fixed by the standard, not tuned.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path
import time
import uuid

from configs import twc_v1_k8_final_config as cfg
import final_runtime as rt
import evaluate_twc_v1_k8_final as base_eval
import evaluate_twc_v1_k8_extended as ext
import evaluate_twc_v1_k8_sscc as sc
import twc_v1_k8_final_results as results

VAL_NAMESPACE = 'twc_v1_k8_typeii_v1'
TEST_NAMESPACE = 'twc_v1_k8_typeii_test_v1'
VAL_BATCHES = 64
TEST_BATCHES = 256
O1 = O2 = 4
SUBBAND_SC = 48                                    # 4 PRBs of 12 subcarriers
# Rel-15 paramCombination 1..6: (L, N_PSK, subbandAmplitude); ordered here from coarse to fine
PARAMS = ((2, 4, False), (2, 8, False), (3, 4, True), (3, 8, True), (4, 4, True), (4, 8, True))
WB_AMP = (0., 1/64**.5, 1/32**.5, 1/16**.5, 1/8**.5, 1/4**.5, 1/2**.5, 1.)
SB_AMP = (1/2**.5, 1.)
VARIANTS = ('fdma_baseline', 'typeii_fdma', 'subband_eigvec', 'perfect_csi_rzf')
CONTRASTS = (('fdma_minus_typeii', 'fdma_baseline', 'typeii_fdma'),
             ('eigvec_minus_typeii', 'subband_eigvec', 'typeii_fdma'))
METRICS = ('sum_rate_bps_hz', 'direction_error', 'interference_power_mean', 'min_user_rate_bps_hz',
           'feedback_bits', 'capacity_bits', 'typeii_L')


def payload_bits(L, npsk, sbamp, nonzero, n_sb, n1=8, n2=4):
    """Rank-1 PMI payload with `nonzero` non-zero coefficients (strongest included)."""
    wideband = math.log2(O1*O2)+math.ceil(math.log2(math.comb(n1*n2, L)))+math.ceil(math.log2(L))+3*(L-1)
    per_sb = (nonzero-1)*(math.log2(npsk)+(1 if sbamp else 0))
    return wideband+n_sb*per_sb


def subbands(nsc=None):
    nsc = nsc or rt.NSC
    return [list(range(s, min(s+SUBBAND_SC, nsc))) for s in range(0, nsc, SUBBAND_SC)]


def dft_beams(grid, device):
    """[O1*O2, N1*N2, Nt] orthogonal beam groups of the oversampled 2D DFT codebook."""
    import torch
    h, v, n1, n2 = grid
    h, v = h.double(), v.double()
    out = torch.zeros(O1*O2, n1*n2, h.numel(), dtype=torch.complex128)
    for q, (q1, q2) in enumerate(product(range(O1), range(O2))):
        for i, (a, b) in enumerate(product(range(n1), range(n2))):
            phase = 2*math.pi*((O1*a+q1)*h/(O1*n1)+(O2*b+q2)*v/(O2*n2))
            out[q, i] = torch.polar(torch.ones_like(phase), phase)/h.numel()**.5
    return out.to(torch.complex64).to(device)


def _nearest(values, table):
    """Index of the nearest table entry in dB (amplitudes); zero maps to the smallest index."""
    import torch
    t = torch.tensor(table, device=values.device, dtype=values.dtype)
    lt = torch.log(t.clamp_min(1e-12)); lv = torch.log(values.clamp_min(1e-12))
    return (lv[..., None]-lt).abs().argmin(-1)


def beam_power(H_dl, beams):
    """Wideband power of every beam of every rotation [B,K,O1*O2,N1*N2]."""
    import torch
    return torch.einsum('qna,bkam->bkqnm', beams.conj(), H_dl).abs().square().sum(-1)


def subband_eigvec(H_dl, sbs):
    """Principal eigenvector of every subband covariance [B,K,Nsb,Nt]."""
    import torch
    out = []
    for sb in sbs:
        Hs = H_dl[..., sb]
        out.append(torch.linalg.eigh(Hs@Hs.conj().transpose(-1, -2)).eigenvectors[..., -1])
    return torch.stack(out, 2)


def typeii(power, vecs, beams, L, npsk, sbamp):
    """Quantized subband precoders w [B,K,Nsb,Nt] (unit norm) and the number of non-zero coefficients."""
    import torch
    top, idx = power.topk(L, dim=-1)                                    # strongest L beams per rotation
    q = top.sum(-1).argmax(-1)                                          # [B,K] rotation
    pick = q[..., None, None].expand(-1, -1, 1, L)
    sel = idx.gather(2, pick).squeeze(2)                                # [B,K,L], descending power
    p = top.gather(2, pick).squeeze(2)
    Bsel = beams[q.reshape(-1)].reshape(*q.shape, beams.shape[1], -1)   # [B,K,N1N2,Nt]
    Bsel = Bsel.gather(2, sel[..., None].expand(-1, -1, -1, Bsel.shape[-1]))          # [B,K,L,Nt]
    wb = torch.tensor(WB_AMP, device=power.device)[_nearest((p/p[..., :1]).sqrt(), WB_AMP)]
    wb[..., 0] = 1.
    nonzero = (wb > 0).sum(-1)                                          # [B,K]
    c = torch.einsum('bkla,bksa->bksl', Bsel.conj(), vecs)              # [B,K,Nsb,L]
    ref = c[..., :1]
    ratio = c/torch.where(ref.abs() > 0, ref, torch.ones_like(ref))
    step = 2*math.pi/npsk
    phase = torch.round(torch.angle(ratio)/step)*step
    amp = wb[:, :, None, :].expand_as(phase).clone()
    if sbamp:
        rel = ratio.abs()/amp.clamp_min(1e-12)
        amp = amp*torch.tensor(SB_AMP, device=power.device)[_nearest(rel, SB_AMP)]
    amp[..., 0] = 1.; phase[..., 0] = 0.
    coef = torch.polar(amp.float(), phase.float()).to(Bsel.dtype)
    w = torch.einsum('bksl,bkla->bksa', coef, Bsel)
    return w/w.norm(dim=-1, keepdim=True).clamp_min(1e-12), nonzero


def expand(W, H_dl, sbs):
    """Per-subcarrier reconstruction g_sb * w_sb with the ideal subband gain."""
    import torch
    H_hat = torch.zeros_like(H_dl)
    for j, sb in enumerate(sbs):
        g = H_dl[..., sb].abs().square().sum(2).mean(-1).sqrt()         # [B,K]
        H_hat[..., sb] = (g[..., None]*W[:, :, j])[..., None].expand(-1, -1, -1, len(sb))
    return H_hat


def capacity_bits(system, sample, gamma, noise, polarity):
    """Bits each UE can send on its 24 FDMA REs (same idealization as the SSCC baseline)."""
    import torch
    phy = rt.physics()
    ce, stats = system['ce'], system['stats']
    H_ul = sample['H_ul']
    batch = H_ul.shape[0]
    pilots = phy.known_pilots(batch, H_ul.device, polarity)
    Y = phy.transmit(H_ul, pilots, noise, gamma)
    est = ce(Y, pilots, phy.fourier_ls(Y, pilots), gamma_db=gamma)
    cevar = stats.ce_variance(gamma)
    index = torch.tensor(stats.layout.data_positions, device=H_ul.device)
    e = rt.ED/index.shape[-1]
    n0 = .25*torch.pow(torch.tensor(10., device=H_ul.device), -gamma/10)
    g = est.abs().square().sum(2).gather(-1, index[None].expand(batch, -1, -1))
    v = cevar.gather(-1, index[None].expand(batch, -1, -1))
    return torch.log2(1+e*g/(n0[:, None, None]+e*v)).sum(-1)            # [B,K]


def typeii_forward(system, sample, gamma, noise, polarity, beams, sbs, grid_dims=(8, 4)):
    """Highest-resolution parameter combination whose payload fits the UE's capacity."""
    import torch
    H_dl = sample['H_dl']
    cap = capacity_bits(system, sample, gamma, noise, polarity)
    power, vecs = beam_power(H_dl, beams), subband_eigvec(H_dl, sbs)
    choice = torch.zeros(cap.shape, dtype=torch.long, device=cap.device)
    Ws, bits_all = [], []
    for j, (L, npsk, sbamp) in enumerate(PARAMS):
        W, nz = typeii(power, vecs, beams, L, npsk, sbamp)
        table = torch.tensor([payload_bits(L, npsk, sbamp, m, len(sbs), *grid_dims) for m in range(L+1)],
                             dtype=cap.dtype, device=cap.device)
        bits = table[nz]
        choice = torch.where(bits <= cap, torch.full_like(choice, j), choice)
        Ws.append(W); bits_all.append(bits)
    W = torch.stack(Ws, 0)
    W = W.gather(0, choice[None, :, :, None, None].expand(1, -1, -1, *W.shape[3:])).squeeze(0)
    bits = torch.stack(bits_all, 0).gather(0, choice[None]).squeeze(0)
    L = torch.tensor([p[0] for p in PARAMS], device=cap.device)[choice]
    return dict(H_hat=expand(W, H_dl, sbs), bits=bits, capacity=cap, L=L.float())


def metrics(out, H_dl):
    import torch
    m = sc.metrics_of(out['H_hat'], H_dl)
    m.pop('dl_nmse_linear')                                             # phase-ambiguous for PMI feedback
    for key, name in (('bits', 'feedback_bits'), ('capacity', 'capacity_bits'), ('L', 'typeii_L')):
        if key in out:
            m[name] = out[key].float().mean(-1).cpu()
    return m


def run(fdma, data, role, batches, gammas, device, grid, sample_fn, stream_fn):
    import torch
    beams, sbs = dft_beams(grid, device), subbands()
    rows = []
    for b in range(batches):
        sample = sample_fn(b)
        for gi, gamma in enumerate(gammas):
            noise, polarity = stream_fn(b*len(gammas)+gi)
            g = torch.full((cfg.MICRO_BATCH,), float(gamma), device=device)
            for name in VARIANTS:
                with torch.inference_mode():
                    if name == 'perfect_csi_rzf':
                        m = metrics(dict(H_hat=sample['H_dl']), sample['H_dl'])
                    elif name == 'typeii_fdma':
                        m = metrics(typeii_forward(fdma, sample, g, noise, polarity, beams, sbs, (grid[2], grid[3])), sample['H_dl'])
                    elif name == 'subband_eigvec':
                        m = metrics(dict(H_hat=expand(subband_eigvec(sample['H_dl'], sbs), sample['H_dl'], sbs)),
                                    sample['H_dl'])
                    else:
                        out = ext.forward(fdma, sample, g, noise, polarity)
                        m = metrics(out, out['H_dl'])
                for j in range(cfg.MICRO_BATCH):
                    rows.append(dict(variant=name, role=role, gamma_db=float(gamma), sample_index=b*cfg.MICRO_BATCH+j,
                                     batch_index=b, **{k: float(v[j]) for k, v in m.items()}))
        if (b+1) % 16 == 0:
            print(f'{role} {b+1}/{batches}', flush=True)
    return rows


def _grid(data):
    import evaluate_twc_v1_k8_channel_shift as cs
    return cs.antenna_grid(cs.bs_positions(data))


def evaluate(mode, device=None, *, loaded=None, data=None, grid=None, out_root=None, batches=None):
    import torch
    import evaluate_twc_v1_k8_test as te
    device = device or rt.base().configure_cuda()
    namespace = TEST_NAMESPACE if mode == 'test' else VAL_NAMESPACE
    root = Path(out_root or Path('runs')/namespace/'reports')
    if mode == 'test' and root.exists() and any(root.iterdir()):
        raise FileExistsError(f'A sealed-test Type II report already exists in {root}; the test set is evaluated once.')
    table, manifest = ext.variants(device, loaded=loaded)
    fdma = table['fdma_baseline'][0]
    if mode == 'test':
        data = data or te.test_data(device)
        batches = batches or TEST_BATCHES
        sample_fn = lambda b: data.sample('test', b)
        def stream_fn(index):
            noise, _, polarity = te.streams(index, device)
            return noise, polarity
        role = 'test'
    else:
        data = data or rt.make_data(device)
        batches = batches or VAL_BATCHES
        sample_fn = lambda b: data.sample('validation', cfg.FINAL_FIRST+b)
        stream_fn = lambda index: rt.batch_random('final', index, device)
        role = 'final'
    grid = grid or _grid(data)
    if (grid[2], grid[3]) != (8, 4):
        raise ValueError(f'Unexpected BS antenna grid {grid[2]}x{grid[3]}')
    started = time.perf_counter()
    rows = run(fdma, data, role, batches, cfg.GAMMAS, device, grid, sample_fn, stream_fn)
    path = root/('attempt_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    base_eval.write_csv(path/'samples.csv.gz', rows, compress=True)
    base_eval.write_csv(path/'summary.csv', results.summarize(rows, role, cfg.GAMMAS, METRICS))
    base_eval.write_csv(path/'comparisons.csv', results.compare(rows, role, cfg.GAMMAS, METRICS[:4], CONTRASTS))
    provenance = dict(namespace=namespace, role=role, batches=batches, variants=list(VARIANTS),
        codebook=dict(release=15, rank=1, O1=O1, O2=O2, N1=grid[2], N2=grid[3], subband_subcarriers=SUBBAND_SC,
                      n_subbands=len(subbands()), param_combinations=[list(p) for p in PARAMS],
                      polarization='single (L coefficients)', cqi='ideal subband gain',
                      feedback_link='capacity of the 24 FDMA REs with the pilot-only UL estimate (no outage)'),
        final_manifest=manifest, code=rt.source_identity(), elapsed_seconds=time.perf_counter()-started,
        training_updates=0, FDMA_updates=0, test_used=(mode == 'test'), selection_on_test=False,
        contains_model_weights=False, contains_channel_tensors=False, torch=torch.__version__)
    (path/'provenance.json').write_text(json.dumps(cfg.plain(provenance), indent=2, sort_keys=True)+'\n')
    (path/'README.md').write_text(f'3GPP Type II (Rel-15, rank 1) baseline on the FDMA frame, role {role}.\n')
    files = {p.name: rt.sha256(p) for p in sorted(path.iterdir())}
    (path/'export_manifest.json').write_text(json.dumps(dict(namespace=namespace, export_id=path.name[8:],
        files=files), indent=2, sort_keys=True)+'\n')
    print('REPORT', path.resolve(), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate', action='store_true')
    mode.add_argument('--evaluate-once-test', action='store_true')
    args = parser.parse_args()
    evaluate('test' if args.evaluate_once_test else 'validate')


if __name__ == '__main__':
    main()
