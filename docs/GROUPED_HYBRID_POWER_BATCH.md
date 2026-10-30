# Independent grouped-hybrid energy-gate continuation

## Scope and evidence boundary

This batch tests explicit exclusive/shared energy adjustment, including a constant-gate control. It does not import another project's communication system, change the decoder, or establish a performance improvement in advance. Only the eleven new power-batch files are added; old models, trainers, physical protocols, configuration fingerprints, results, and the manuscript remain unchanged.

The user reports that the six runs in `runs_as_screen/grouped_hybrid_anchor_v1/batches/batch_cf66957ea2914720bbdd9ed15dbf3884/` completed 20,000 updates, with best/latest at 20,000 and prior counts 20,000. These server artifacts have **not** been read locally. The controller checks their exact records, completed stages, immutable evidence, archived repair chain, evaluation inputs/outputs, and checkpoint identities before new training. It neither discovers a different batch nor retrains the old six runs. The original source identity at `266d9e5` and the repair at `ad4e801` remain distinct; archived 500-step active hashes are not incorrectly compared with completed 20k weights.

## Fixed arms and sources

Both training seeds are `20261030` and `20261031`. Each of four arms gets a new optimizer/scheduler and exactly 20,000 additional updates:

| New arm | Exact same-seed anchor source | New parameters |
| --- | --- | ---: |
| `fdma_control` | `fdma_control` best | 0 |
| `h32_control` | `hybrid_f32_a64` best | 0 |
| `h32_static_power` | `hybrid_f32_a64` best | 1 |
| `h32_gamma_power` | `hybrid_f32_a64` best | 6 |

Pinned **best.pth** SHA256 values (not filename guesses):

| Source / seed | SHA256 |
| --- | --- |
| FDMA / 20261030 | `eb8a4514ba8fe6b2e57472c1a7c2a6bd8259855e394fb6310f8901241c42c26d` |
| FDMA / 20261031 | `79b399b13125abc6857b7d851c9e11a80a7f2c093f061353fe878e5802de213e` |
| H32 / 20261030 | `aae67b5009a22229d0133d96ab4d10e8f8015b7fcb3319c6801ae3a439af8c76` |
| H32 / 20261031 | `331a52267b376656d4c6bbf6d69e98e3bdc950137a54d68d5d7ecb39c4f150e1` |

Paths come from the unmodified anchor configuration and are cross-checked against the fixed batch. Source configuration, best/latest states and steps, validation selection, prior state, and sidecar hashes are checked. Earlier grouped-codec F weights are not accepted. Metadata uses canonical JSON comparison; source-code and weight hashes remain exact.

## Only the energy gate changes

K4, Nt32, Nsc256, perfect UL CSI, 256 feedback RE, DL25 and per-UE total energy64 are unchanged. H32 uses 32 exclusive coordinates per UE and 64 shared coordinates per two-user group: `4*32+2*64=256`, 96 active symbols per UE. FDMA uses 64 exclusive symbols per UE.

The six gamma-gate scalars correspond to `[0,5,10,15,20,25]` dB. Linear interpolation produces `a(gamma)` and

```
g = log(4) * tanh(a)
exclusive_raw *= exp(g/2)
shared_raw unchanged
original joint normalization to energy64
```

The relative exclusive/shared energy ratio multiplier is bounded by `[1/4,4]`. The static arm uses one gamma-independent scalar with the identical transformation. Gates are shared by all four UEs and see only public gamma. All scalars start at zero, consuming no random draws. There is no independent branch normalization, extra loss, or special gate learning rate. Gate parameters are included in the UE parameter count; `full = UE + BS`, so the gate is not counted twice. Absolute model counts require actual server instantiation and are recorded by preflight, training inspection, and evaluation.

The existing Legacy UE and complete joint/Swin BS decoder remain. Only the original exclusive raw codeword coordinates are multiplied. Physical placement, noisy reception, historical-prior regularized recovery, full angular-delay representation, inverse transform, normalized RZF, and task sum-rate path are retained. No actual transmitted codeword or downlink truth is fed into the receiver; truth is detached for offline diagnostics. H32 cannot become full FDMA64 merely by reducing shared energy: it still has only 32 exclusive coordinates.

`N0 = 0.25 * 10**(-gamma/10)` uses the transmit-energy reference, **not** historical received feedback SNR. No old Swin curve is imported. A Perfect-CSI RZF reference is optional in the task and is not included in this implementation; the evaluator contains exactly six prescribed methods.

## Two clocks and strict continuation

All source `pi_nodes`, `update_count`, and `node_update_mass` are migrated unchanged. Strict source-state migration allows only the new zero gate parameter, then uses `strict=True`. Normal new-checkpoint loading also uses `strict=True` and never reinitializes the gate.

| Local optimizer/scheduler step | Committed cumulative prior count |
| ---: | ---: |
| 0 | 20000 |
| 500 | 20500 |
| 20000 | 40000 |

The execution adapter addresses data, gamma, and standard noise using the **local** step in `grouped_hybrid_power_v1`. It passes `20000 + local_step` to the model's inherited prior API. Forward reads the previous committed prior. The lifecycle commits actual gated, jointly normalized codeword statistics only after a successful optimizer update. Validation/test never commit priors. No Sionna private generator is serialized into checkpoints.

The unmodified `main_train.train_proposal` handles training through existing model-factory, optimizer-factory, execution, lifecycle, and replayable-data interfaces. No monkeypatch or second trainer is introduced. New initialization strictly migrates source weights before optimizer creation; resume validates full new configuration/fingerprint, source identity, code hashes, local and cumulative counts, model tensors, optimizer moments/order, scheduler, and complete default RNG state. Step0 competes in checkpoint selection and may remain best. Full interrupted-versus-uninterrupted training bitwise equality has not been established.

The inherited current recipe is batch48, common UE/BS/gate peak LR `1e-4`, AdamW weight decay `1e-5`, clip `0.5`, OneCycle warmup10%, all auxiliary weights zero, uniform gamma0–25 and DL25. The current anchor recipe uses **256 fixed validation channels in chunks of8**, not a stale 64-channel recipe. Every500 updates it evaluates all six gamma nodes and selects a single best by `mean_all`. The first500 are part of the20k scheduler; automatic resume starts at501 without resetting it. FP32/complex64, highest matmul precision, TF32 off and existing activation checkpointing are retained. There is no automatic microbatch, precision, seed, or budget fallback.

## Controller and checks

All new products are under `runs_as_screen/grouped_hybrid_power_v1/`. Default CLIs display help; dry-runs are standard-library-only plans that do not read checkpoint bytes or create result directories. Explicit actual runs require CUDA. The controller holds a dedicated lock; each training run has its own lock and isolated directories. Existing completed stages are reused only if their recorded evidence hashes still match. Resume requires the same code identity. Failure retains a report and exits nonzero rather than repairing data or clearing directories.

The batch executes:

1. Environment and exact old-batch/source verification.
2. One new gate-specific CUDA preflight: common initialization, exact migration/key/shape/dtype checks, actual parameter counts, zero-gate transmitted/received tensors, H_hat, normalized RZF and rate at gamma0/12.5/25, then physical batch48 forward/backward/scratch-update memory checks for all four arms. Numeric comparisons use preset `rtol=atol=1e-6`; errors are reported, never silently relaxed. Scratch checks do not save weights or add official training updates.
3. Eight new500-update pauses with automatic continuation to20k; source data and old outputs remain read-only.
4. Two independent six-method evaluations and CSV-only result organization.

The controller does not rerun the already accepted Full/Lite/RNG suites, delete old runs, or connect to a server itself. Server execution is initiated only by the user.

## Evaluation and interpretation

New test seed `20270108` and the new namespace are checked against the current training/validation and prior anchor test namespace. Each seed is evaluated separately: four continued arms plus frozen completed FDMA and H32 sources. Each of six gamma conditions uses1600 complete multi-user channels in200 batches of8. The methods share actual H_dl/H_ul and standard-noise tensors; input hashes are saved and checked across both training-seed evaluations. Checkpoint files, model parameters/buffers, execution mode, and priors are checked unchanged. All negative differences remain.

Eight predeclared contrasts are gamma–H32, static–H32, gamma–static, gamma–continuedFDMA, static–continuedFDMA, H32–continuedFDMA, continuedFDMA–frozenFDMA, and H32–frozenH32. Each point, low0/5/10 average, and full-six average is computed from paired samples; cross-gamma averaging happens within each complete channel before computing sample and batch intervals. Batch unit is `sample_index//8`; sample and batch SE use `ddof=1` and CI=`mean±1.96SE`. A one-batch evaluator preflight has unestimable batch intervals, not zero intervals. It is not included in formal results or used to select weights.

The engineering screen against **continued FDMA** requires low-three mean gain≥0.5, batch CI lower>0, and gamma25 batch CI lower≥−0.1. Improving H32 alone does not satisfy this screen, and claiming gamma adaptation requires comparison with the independently trained static gate. These thresholds are not journal acceptance criteria. Neither seeds nor historical and current protocols are pooled. CI measures evaluation sampling conditional on checkpoints, not training-seed stability.

Outputs include36-condition summary, raw samples CSV.gz,64 signed paired summaries, raw paired samples, actual parameters, theta/gates, energy/per-active-symbol energy, soft/corrected codeword errors and uncalibrated prior-based proxy, downlink desired/interference diagnostics, input hashes and manifest. Per-subcarrier RZF power is1; the legacy full-band power metric is256. Raw NMSE is secondary, not a checkpoint-selection objective. No post-training gate-disable intervention is presented as a retrained ablation.

## Results and user commands

The batch stores `batch_summary.json`, exact evaluation pointers, source/preflight/selection evidence, and a unique small ZIP. The export contains all six methods' six-point curves (separate panels for training seeds), signed batch-CI plots, gate/energy curves, validation curves, last5k trends, CSV/JSON and English/Chinese internal draft paragraphs. Selecting step20k does not establish convergence. Weights and compressed per-sample data remain outside the small ZIP on the server. No performance numbers are populated from chat or written to the manuscript.

After safely syncing this commit and checking for existing processes/locks, the single actual batch command is:

```bash
python -B -u run_grouped_hybrid_power_batch.py --run
```

Explicit interruption recovery (same code, no live owner) is:

```bash
python -B -u run_grouped_hybrid_power_batch.py --run --resume
```

The exact compact feedback command, including live small-file progress, is:

```bash
python -B run_grouped_hybrid_power_batch.py --summary
```

It prints status/errors, stages, eight runs and selected steps, actual parameter counts, gates and branch energies, both seeds' complete six-point rates, signed sample/batch intervals, engineering decisions, last-five validation records/last5k trends, and the exact result-package path. Unknown progress remains explicitly unknown; summary does not load weights.

## Local verification record

Executed locally on 2026-10-30: `python -B -m unittest tests.test_grouped_hybrid_power tests.test_grouped_hybrid_power_batch -v` ran34 tests: **33 passed,1 explicit server-only skip,0 failures/errors**. Coverage includes tiny real-Torch gate gradients and physical normalization/recovery, cumulative-prior500-to501 behavior, canonical source identity, strict resume schemas, completed-stage reuse, failed-smoke/output protection, summary rendering, paired covariance/batch statistics, and rejection of incomplete/nonfinite results. These are not fabricated server checkpoints or dependency substitutes.

All ten new Python files passed AST syntax checks. Trainer/evaluator dry-runs are covered by lightweight CLI tests; controller/preflight `--dry-run` and controller `--summary` also passed with `python -B -S` (no site packages). No new experiment output root was created. The full model/checkpoint/CUDA test is explicitly skipped locally. No real source checkpoints, CDL batches, full models, training, GPU preflight, formal evaluation, figures from experiments, or SSH sessions have been run here. Actual server source acceptance, zero-gate full-network errors, physical batch48 memory/gradients, eight new trainings, and all performance results remain unverified until the user runs the batch.
