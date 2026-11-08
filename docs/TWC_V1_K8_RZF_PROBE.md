# Frozen-checkpoint K8 RZF diagnostic

## Status and scope

Prepared on 2026-11-08. The user reports completion of the four task-loss arms, each with 6048 new updates and cumulative update 22096. That completion is a user-provided fact; no server checkpoint or final performance result is claimed to have been read locally for this preparation. The present diagnostic has not yet produced a selected alpha or a server validation result.

The only varied computation is the scalar regularization alpha passed to the existing RZF solver and its existing per-subcarrier power normalization. All four codecs, the CE, physical feedback/noise convention and task-loss checkpoint states remain frozen. There is no training, optimizer, gradient-scale beta calibration, architecture change or checkpoint selection. In particular, this diagnostic does not rewrite the preceding training recipe, beta or its evidence.

New outputs are isolated under `runs/twc_v1_k8_rzf_probe_v1/`. Source experiment implementation `596d2a4d640c9001b85f78dec939bf032ff4a024`, ordered-window source `b6dc7ed5453ad0f8bc581b157505e4656932c1b7` and the internal schedule-screen provenance `8096ed965027a92f6e0738c7653b4c3e75f71b8c` remain separate links. The new diagnostic commit is recorded separately; a different current HEAD is not substituted for any historical source version.

## Four pinned final sources

Paths are relative to the 03 repository root. These are prescribed final checkpoints, not a search for the newest or highest-scoring file.

| Method | Checkpoint | SHA256 |
| --- | --- | --- |
| `fdma_nmse` | `runs/twc_v1_k8_task_loss_screen_v1/training/fdma_nmse/checkpoint_latest.pt` | `5feae0f5c3d6da9386a05d6c4ff1a5d939c54a7a1ebf2ea7065e23fa46dd13be` |
| `fdma_joint` | `runs/twc_v1_k8_task_loss_screen_v1/training/fdma_joint/checkpoint_latest.pt` | `fd1927f6748b5eee35e2efd42e37146b5ab7cc791130bad93b2b8c96966e2926` |
| `g4_nmse` | `runs/twc_v1_k8_task_loss_screen_v1/training/g4_nmse/checkpoint_latest.pt` | `6fa136d9ce5afbc545a85d1646018dbd609df2a9b2a4e2524c42aa7c4fb67d57` |
| `g4_joint` | `runs/twc_v1_k8_task_loss_screen_v1/training/g4_joint/checkpoint_latest.pt` | `c2fbfd3e1508ea7983b618ec0a863b455dd540e051d6f2e6469779d6951adfe1` |

The server verifies checkpoint bytes, complete saved configuration/fingerprint, final local step 6048, cumulative step 22096, training completion, model identity, CE and the source provenance chain before running a model. Model loading is strict; trained ordered-window weights are neither migrated again nor reset. Checkpoints and their original manifests/logs remain read-only and are not copied into the new diagnostic output tree.

The system setting remains K8/Nt32/Nsc256, the original 256-RE protocol, frozen CE, DL SNR25, six gamma points 0/5/10/15/20/25 dB and Ep16. Hybrid remains G4 and reports both key:shared ratios 1:3 and 1:1. Existing energy/noise definitions and actual resource mapping are retained.

## Alpha selection and independent final validation

The fixed reference is alpha 0.01. The predeclared search grid is:

```
[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
```

Selection uses only 64 complete multiuser validation channels: 16 batches of four at fresh `validation` addresses 4096–4111. For each checkpoint, rank alphas by mean downlink sum rate over all six gamma conditions. For a G4 checkpoint, both energy ratios participate with equal weight, so its score is the equally weighted mean of 12 condition means. FDMA has six equally weighted conditions and is not duplicated to create an extra ratio sample. Exactly equal scores choose the smaller alpha; no tolerance-based tie rule is introduced. Each checkpoint gets exactly one global selected alpha; no per-gamma, per-ratio or per-sample selection is allowed.

All nine grid scores, their ranking, the chosen alpha and a flag for a grid-edge choice are retained, including unfavorable outcomes. The selection stage records 324 condition-mean rows and 36 global-alpha score rows: 36 physical conditions × nine alphas, then four checkpoint objectives × nine alphas. Selection is written to a non-overwriting record bound to source, configuration, input and code identities. Resume reuses this locked record after verification; it must not silently rescore, enlarge the grid or replace the selected alpha after final validation.

The final comparison uses a separate 256 complete multiuser validation channels: 64 batches of four at addresses 5120–5183. For each frozen checkpoint, compare the fixed alpha 0.01 against its previously locked global selected alpha. These final channels cannot affect alpha selection. Test remains sealed. No best-checkpoint choice, gamma-specific oracle or feedback-SNR subset is derived from these results.

## One neural forward per condition

For each batch/method/physical condition, compute H_hat once from the noisy feedback path and reuse that same in-memory H_hat for all nine RZF alphas. Final validation similarly reuses one H_hat for fixed-versus-selected RZF. The protocol requires 576 neural forwards for selection and 2304 for final validation; the nine-alpha grid does not multiply those neural-forward counts by nine. Changing alpha must not trigger another model transmission, draw another feedback-noise realization or change the neural reconstruction. H_hat is not saved as a persistent cache.

The neural model and CE stay in eval/no_grad. Parameters, buffers, modes and input checkpoint hashes are checked before/after. Common actual H_dl/H_ul and pilot/data standard noise establish pairing. Fixed-versus-selected records share the same raw NMSE and H_hat/true-channel norm diagnostics by construction; any alpha-dependent change in those fields is a diagnostic failure rather than a new reconstruction result.

Real H_dl is used only after reconstruction for offline scoring, raw reconstruction-error/norm diagnostics and sum-rate evaluation. It is not supplied to the neural decoder or used to fit a per-sample/user/subcarrier rescaling of H_hat. No oracle amplitude correction is deployed.

The diagnostic also records the scale-relative regularization

```
alpha_relative = alpha / (trace(H_hat H_hat^H) / K)
```

at the relevant sample/subcarrier, with its aggregation explicitly identified. It is a recorded quantity, not a new sample-dependent solver setting: the solver still receives the locked absolute alpha. Power, signal and interference diagnostics remain in their original physical units.

## Interpretation and paired statistics

With the conventional unnormalized RZF form `H^H (H H^H + alpha I)^-1`, replacing H by one global nonzero complex scalar `c*H` is equivalent to using alpha divided by `|c|^2`, up to a scalar that disappears under power normalization except for a common phase. This is a global-scale identity. It does not justify fitting user-specific, subcarrier-specific or sample-specific scales using the true channel. A reduction in the Hybrid–FDMA gap after alpha adjustment would indicate sensitivity to the deployed regularization; it would not prove that amplitude alone explains the original gap or that the reconstruction error is solved.

Preserve all six gamma points, both G4 ratios, signed gains/losses and the fixed-alpha baseline. The final output contains 72 fixed/selected-policy conditions, 18432 raw sample rows and 640 paired summary rows. Report fixed-versus-selected improvements within each checkpoint, Hybrid–FDMA comparisons under the same declared policy, and NMSE-versus-joint comparisons without selecting the more favorable reporting policy afterward. Across-gamma summaries first average within each complete multiuser sample. Calculate both sample-level and batch-level paired SE/95% intervals from raw paired differences; do not subtract CI endpoints or count users/subcarriers/gamma rows as independent extra channels. Batch size is four. Intervals describe evaluation sampling conditional on the fixed checkpoints and this fixed calibration selection. They quantify neither multiple-training-seed stability nor uncertainty in selecting alpha from a different calibration set.

## Existing training trends

Read the four completed task-loss `progress.jsonl` files without changing them. Preserve their source paths/hashes and separate training NMSE, unscaled rate, beta and combined loss. These are different quantities, especially during the 756-update beta warmup.

Report the last four monitor records and the recent trend per monitored condition. The fixed historical monitor uses the original RZF alpha 0.01 and a small validation set at gamma10/Ep16; for G4 it uses ratio 1:3. It is not the six-gamma/two-ratio selection score and is not recomputed with the selected alpha. Last-four-point changes or selecting a final 6048-step checkpoint do not establish full-SNR convergence, and this diagnostic does not rule out a separately authorized future continuation. No new updates are scheduled automatically.

## Controller and result handoff

The single server controller performs source verification, alpha selection, final validation, existing-training trend extraction and compact result packaging. Necessary identity, finite-value, freezing or pairing failures preserve evidence and terminate with a nonzero exit. Completed stages and the immutable selection record are reused only after their hashes and identities are checked. A completed selection-stage manifest that has not yet been sealed may be sealed once from its existing verified summary, without new model inference. A partial calibration is not automatically reselected. A failed final-validation attempt may be retried in a new directory while retaining the old failure receipt and the same locked alphas. An incomplete selection is never silently treated as a valid lock.

In the existing CUDA/Sionna environment, from the 03 repository root:

```bash
python -B -u run_twc_v1_k8_rzf_probe.py --run
```

Read the complete summary, without loading weights:

```bash
python -B run_twc_v1_k8_rzf_probe.py --summary
```

Explicit continuation of an interrupted diagnostic:

```bash
python -B -u run_twc_v1_k8_rzf_probe.py --run --resume
```

The evaluation entry point's internal selection/evaluation modes are invoked by the controller; the user need not start them separately. Summary exposes stage/error state and unknown values while incomplete, the nine-grid scores and four chosen alphas with edge flags, all six-point final results, paired intervals, last-four monitor records and exact result-package path. Source weights and raw historical evidence remain at their original paths. No paper, model, old trainer or result is changed.

## Actual local checks and remaining server work

Executed locally on 2026-11-08: `python -B -m unittest tests.test_twc_v1_k8_rzf_probe -v` ran 19 tests, all passed with 0 failures/errors/skips. An independent `-B -S` run passed the 16 standard-library checks and explicitly skipped three numerical checks because the real Torch dependency was unavailable in that mode. Four new Python files passed AST syntax checks and both evaluation/controller entry points passed standard-library dry-runs without creating the new output root. The 30 pre-existing source files remain outside this batch's edit scope.

The small synthetic check calls the original RZF solver on a B1, K8/Nt32/Nsc256 channel tensor; it does not instantiate a neural codec. Global scaling by 0.5 and 2 gave maximum absolute error 0 in the corresponding scale identity. Scaling by 1.7 gave maximum absolute error `1.20137e-7` and maximum relative error `4.31356e-5`. It passed the predeclared elementwise `allclose` check with rtol 2e-5 and atol 2e-6. This does not assert that the maximum relative error alone is below rtol: elementwise acceptance includes the absolute-tolerance term.

Local work does not load source weights, instantiate the full codec/CE, generate CDL, connect SSH, run training or run GPU/model evaluation. Missing numerical dependencies are reported honestly rather than installed or fabricated. No selected-alpha record has been generated locally.

Real strict source loading, frozen-model forward, common-input verification, the locked alpha selection, independent final validation and true historical-progress parsing remain server work. No alpha, performance benefit or convergence conclusion is reported before those artifacts exist.
