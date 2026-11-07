# K8 task-loss controlled screen

## Status and question

Prepared on 2026-11-07; local verification completed on 2026-11-08. The user reports that all four ordered-window v2 screening arms finished their prescribed 3024 new updates, reaching cumulative update 16048. This is a user-provided completion record; this preparation does not claim the server weights or final result package were read locally. No task-loss calibration, training or validation result has been obtained for this new batch.

The question is whether a controlled task-oriented loss change improves the ordered encoder's downlink performance, including its Hybrid–FDMA gap. The comparison changes the training objective only. The ordered v2 encoder, Joint BS decoder, CE, physical feedback protocol, branch-energy normalization and deployed normalized RZF remain unchanged. A Hybrid improvement by itself is insufficient: the corresponding FDMA loss change is a matched control.

The output namespace is `runs/twc_v1_k8_task_loss_screen_v1/`. Existing schedule-screen and ordered-window v1/v2 outputs remain read-only. The source implementation commit `b6dc7ed5453ad0f8bc581b157505e4656932c1b7` and its earlier internal schedule-screen provenance `8096ed965027a92f6e0738c7653b4c3e75f71b8c` are separate identities. The new experiment commit is recorded separately and must not replace either source identity.

## Four arms and fixed budget

Both layouts use the corresponding completed ordered v2 final checkpoint as their source:

| Layout | Source arm | New objective arms |
| --- | --- | --- |
| FDMA | `ordered_fdma`, final local step 3024 / cumulative step 16048 | `fdma_nmse`; `fdma_joint` |
| Hybrid | `ordered_g4`, final local step 3024 / cumulative step 16048 | `g4_nmse`; `g4_joint` |

Pinned source files, relative to the 03 repository root:

| Source | Path | SHA256 |
| --- | --- | --- |
| FDMA | `runs/twc_v1_k8_ordered_encoder_screen_v2/training/ordered_fdma/checkpoint_latest.pt` | `25176c8d770775891fa928a60607054b64d91a2049856da626031a7451fd5203` |
| G4 | `runs/twc_v1_k8_ordered_encoder_screen_v2/training/ordered_g4/checkpoint_latest.pt` | `1a7ac64e45964043e23af96a69bbdf4bd144352e8cd6708f6dcca15eda4581a4` |

The server must verify source manifests, hashes, model/configuration identity, final step and provenance before loading them. The complete trained ordered v2 state, including its learned window-residual weights, is loaded strictly; the old mean-to-ordered migration is not rerun and the residual weights are not reset. The controls and joint-loss arms within a layout start from identical model state. No source optimizer or scheduler state is continued: all four arms create a fresh AdamW and constant scheduler with the same recipe.

| Item | Fixed setting |
| --- | --- |
| Geometry | K8 / Nt32 / Nsc256 / 256 feedback RE |
| Layout | Existing FDMA or Hybrid with G4, unchanged physical mapping |
| New updates | 6048 per arm; 24192 total |
| Training seed / batch | 20261107 / 4 complete multiuser channels |
| Optimizer | Fresh AdamW; LR 2e-4; weight decay 1e-5 |
| Scheduler / clipping | Constant; gradient norm clipped at 1 |
| CE | Frozen parameters, eval mode and normalization statistics |
| Precoder | Existing normalized RZF; alpha 0.01, unchanged |
| Data addresses | `codec_train`, cumulative steps 16049–22096 |
| Smoke segment | First 108 updates are included in the 6048-update budget |
| Selection | Prescribed final checkpoint only; no test- or SNR-specific best selection |

The existing gamma, pilot-energy and key/shared-energy schedules continue at the cumulative address. G4 stays fixed rather than cycling G1/G2. Corresponding new updates share the same actual channels and standard pilot/data noise across arms. Existing transmit-energy and noise conventions are not reinterpreted as a different feedback SNR protocol.

## One shared gradient-scale calibration

Calibration is completed before any new optimizer update and is immutable once accepted. It uses twelve fresh `codec_train` addresses 500000–500011, each a batch of four complete multiuser channels, for each of the two source layouts. Each of the six gamma values appears twice; Ep is fixed at 16. The Hybrid source uses G4 and key:shared energy 1:3. These are calibration records, not final validation/test observations.

Both source codecs remain in eval mode while gradients are enabled. Calibration computes the NMSE and deployed-rate gradients without creating or stepping an optimizer. The gradient norm includes the complete codec parameter set; a missing gradient contributes zero rather than silently changing the parameter basis. Its squared-gradient norm accumulation uses FP64 for this diagnostic scalar only; model forward/backward and gradient computation retain their original dtypes. CE remains frozen. After each record, gradients are cleared and model parameters/buffers are checked against the frozen source state.

For the combined 24 layout/batch records, let `g_N` and `g_R` be the recorded full-codec gradient norms of NMSE and negative downlink mean rate. The fixed rule is:

```
beta = 0.25 * median(g_N) / median(g_R)
```

The same scalar beta is used by both joint-loss arms; it is not separately tuned for Hybrid or FDMA. Nonfinite, zero-denominator or incomplete calibration evidence is rejected. The record preserves the individual norms and source/data/code identities, not only the scalar result. No beta value is claimed before actual server calibration.

For a joint-loss arm at new local update `step`, the loss is exactly:

```
L = L_nmse - beta * min(step / 756, 1) * mean_downlink_rate
```

The mean rate uses the existing deployed RZF path. There is no Rref, new rate normalization or change of loss scale beyond the prescribed beta. The first 756 warmup updates are included in the 6048-update budget. The NMSE controls retain their original objective. Calibration identity and its source/data/configuration fingerprint are required for initialization and strict resume; an existing accepted calibration cannot be silently regenerated or replaced. Changing beta, the warmup, source or any required calibration identity requires a different experiment identity.

## Validation and signed paired comparisons

The independent screening validation uses exactly 256 complete multiuser samples, 64 batches of four, at fresh `validation` addresses 3072–3135. It does not use the calibration addresses, the preceding ordered-screen validation addresses or the sealed test split.

Conditions are gamma 0/5/10/15/20/25 dB, Ep16 and downlink SNR25. Hybrid remains G4 and reports key:shared energy ratios 1:3 and 1:1 separately. FDMA is evaluated once per gamma, not repeated as artificial additional samples for the two Hybrid ratios.

Six fixed checkpoints are evaluated on common actual H_dl/H_ul and common pilot/data standard noise: the four new final checkpoints plus the two unchanged ordered v2 source checkpoints. This produces 54 conditions: three FDMA methods × six gammas, and three Hybrid methods × six gammas × two ratios. Saved input fingerprints establish pairing beyond a shared seed.

Report all gamma points and both Hybrid energy ratios for DL NMSE, downlink sum rate, desired-signal/interference power, energy checks and per-subcarrier RZF power. The eight per-user rate columns remain secondary diagnostics within the same complete multiuser sample. Preserve signed differences, including negative outcomes. The principal comparisons are joint loss minus NMSE within each layout, and the difference in the Hybrid–FDMA gap between those objectives. Also compare each new arm against its matching frozen source. Source and new-arm conditions must be explicitly paired; shared FDMA records are references, not extra independent observations.

For across-gamma summaries, average the prescribed gamma rates within each complete multiuser sample before calculating a paired mean, SE and 95% interval. Preserve both sample-level and batch-level paired intervals; batch size remains four complete multiuser channels. Do not average separate SE values, subtract CI endpoints or treat users/subcarriers/gamma rows as independent extra samples. This is a single-training-seed validation screen, not a claim about training-seed stability or final journal test performance.

## Controller, recovery and outputs

The server controller performs source verification, calibration, four 108-update technical segments and their automatic continuation to 6048, common validation and result packaging. A successful technical segment is not an additional training budget and does not reset optimizer, scheduler or data addresses. All necessary identity/numerical checks must pass before continuation; a failure retains its receipt and ends the batch.

Explicit resume validates complete model, optimizer, scheduler and RNG state, local/cumulative steps, source and CE identities, immutable calibration and its fingerprint, configuration and code identity. Checkpoints retain the full calibration record, its content identity and file SHA256. An accepted calibration is reused read-only; a retained attempt without a completed final calibration record is rejected rather than automatically recalibrated after an interruption. Resume must not fall back to initialization or regenerate calibration when required evidence is missing. Completed stages are reused only after their recorded files and hashes pass verification. No historical checkpoint, raw sample file or manifest is rewritten to make a source compatible.

The batch retains checkpoint/configuration identities, calibration records, progress and objective components, final checkpoints, complete validation summaries/raw samples, sample/batch paired statistics and compact CSV/JSON outputs. The small result package is a verified ZIP; this batch does not generate figures or result-paragraph drafts. Its summary distinguishes calibration, training and validation status and exposes unknown information during an incomplete run. Weights and full raw samples remain at the server; the small result package excludes them. No new numerical performance claim is written into the existing paper from this code preparation.

From the repository root in the existing CUDA/Sionna environment, the single server controller is:

```bash
python -B -u run_twc_v1_k8_task_loss_screen.py --run
```

Read the complete progress/result summary without loading weights:

```bash
python -B run_twc_v1_k8_task_loss_screen.py --summary
```

After an interruption, explicitly resume the same batch:

```bash
python -B -u run_twc_v1_k8_task_loss_screen.py --run --resume
```

The controller performs calibration internally; the user does not need to start the training entry point's calibration/report modes separately. Default entry-point invocation shows help; `--dry-run` is standard-library-only and performs no model/checkpoint loading or result-directory creation. Necessary identity or numerical failures stop the batch and preserve evidence.

## Local verification and remaining server work

Executed locally on 2026-11-08: `python -B -m unittest tests.test_twc_v1_k8_task_loss_screen -v` completed 24 tests, all passed with 0 failures/errors/skips, including three small Torch tensor numerical checks. An independent `-B -S` run completed the same 21 standard-library checks and explicitly skipped the three Torch-dependent checks because site packages were unavailable in that mode. All five new Python files passed AST syntax checks. The training, evaluation and controller entry points each passed a `-B -S --dry-run`; the independent task-loss output root was not created.

These checks cover the loss/warmup and calibration identities, calibration gradient-norm handling and shared-beta rule, immutable/incomplete calibration refusal, source/resume metadata, final-step selection, output protection and signed paired statistics. They do not load a real source model or checkpoint. No CDL generation, large-model forward/backward, training, GPU evaluation or SSH connection is performed locally. No large dependency installation or fabricated Sionna module is used.

Server verification remains required for real source identity/loading, deployed-rate gradients and beta calibration, CUDA memory/numerical behavior, the four new training arms, strict recovery and all validation statistics. The preceding ordered v2 completion is a user-provided fact; it is not new task-loss evidence. Test remains sealed and Stage C is not implicitly started under an old configuration or fingerprint.
