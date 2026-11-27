# K8 review experiments: key-count ablation and geometry study

Status 2026-11-27: code and CPU tests done; not yet run on a GPU. Nothing below is a result.

Both experiments answer reviewer-style objections to v6 of the paper and run on the H100,
where the shared CE (`~/precoder_shared_ce/nmse65000`) and the q = 4 reference `g4_joint`
(`runs/twc_v1_k8_decoder_cmp_v1/g4_joint`) already exist. The frozen K8 FDMA (new machine)
is neither needed nor changed.

## 1. Key-count ablation (`twc_v1_k8_key_ablation_v1`)

Question: with the same pairs, are dedicated key REs needed, or would pure sharing do as well?

| arm | key REs/UE | shared REs/pair | data dims/UE |
|---|---:|---:|---:|
| q0 | 0 | 48 | 48 |
| q2 | 2 | 44 | 46 |
| g4_joint (reference, existing) | 4 | 40 | 44 |
| q8 | 8 | 32 | 40 |

Same recipe, codec initialization and training channel/gamma stream as g4_joint
(group-size recipe, 30k updates from scratch, fixed consecutive pairs, CE frozen). Only q
differs. `keyed_runtime.py` generalizes the layout, waveform, error-aware LMMSE receiver and
decoder geometry to any q; q = 4 reproduces grouped_runtime exactly (CPU test).

Evaluation: `--evaluate` on the final model's validation frames (same as every grouped
report), then `--evaluate-once-test` once on the K8 sealed test frames/noise/polarity
(namespace `twc_v1_k8_key_ablation_test_v1`, with batch_means.csv for pairing with FDMA
and g4_joint test reports).

## 2. Geometry study (`twc_v1_k8_geometry_v1`)

Question: does the gain survive when users have different geometry and the models are
trained for it?

FDMA (24 REs/UE) and the paired Hybrid (4 key + 40 shared) are trained from scratch with
the same recipe, network and initialization. Each codec-training MU of CDL-C draws
D from {0, 1, 2, 4} and gives each user an independent angular offset, uniform in
[-D, D] DFT bins per array axis, as a phase ramp on the user's DL and UL channel (the
rot<D> model of the channel-shift report). This trains a new FDMA model (authorized by
the user on 2026-11-27); recipe field `new_FDMA_model_updates` records it, and the
manifest field `FDMA_updates` of the shared trainer refers to the frozen K8 FDMA.

Evaluation (`--evaluate`): rot0/1/2/4/8 with the channel-shift report's frames, offsets
and noise (unseen in training): fdma_geo, hybrid_geo (fixed pairs), hybrid_geo_paired
(least-coupled pairs from an independent sounding), g4_joint_cdlc (CDL-C-only reference)
and perfect_csi_rzf, plus UL NMSE of the frozen CE and the true UL coupling.

Caveat to report with the results: the CE was trained on CDL-C without offsets and stays
frozen for both arms; the UL NMSE column shows how much it degrades.

## Machine plan (updated 2026-11-27 21:00)

- H100: fdma_geo and hybrid_geo (started 11:48, 5 processes at first; q0, q2, q8 stopped at
  ~1k updates at ~14:00 because the machine's total throughput is fixed at ~0.69 updates/s).
  After the geometry terminals: geometry evaluation, then q8 resumed from its checkpoint.
- New machine (RTX 5090, account rong): q0, then q2, from scratch, one after another
  (`tools/queue_twc_v1_k8_key_ablation_new.sh q0 q2`), each followed by its validation report.
  These are the runs that count for q0 and q2; the partial H100 directories of q0 and q2 are
  left untouched and never resumed or evaluated.
- Training channels are generated on each machine from the same seeds; across machines they
  agree up to float32 rounding (determinism check of 2026-11-20).

## Commands (H100)

    bash tools/start_twc_v1_k8_review_runs.sh fdma_geo hybrid_geo
    bash tools/start_twc_v1_k8_review_runs.sh q8              # after the geometry terminals

## Commands (new machine)

    bash tools/setup_shared_ce.sh            # check; add --copy if the shared CE is missing
    bash tools/queue_twc_v1_k8_key_ablation_new.sh q0 q2

After the terminals exist:

    python -B train_twc_v1_k8_key_ablation.py --arm q0 --evaluate
    python -B train_twc_v1_k8_key_ablation.py --arm q0 --evaluate-once-test
    python -B train_twc_v1_k8_geometry.py --evaluate
