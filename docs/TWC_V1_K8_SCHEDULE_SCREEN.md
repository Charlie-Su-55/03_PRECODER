# K8 condition-schedule screening batch

## Question and evidence boundary

This is a limited continuation experiment for the completed TWC v1 K8 NMSE warmup. It tests whether short-term adaptation to the most recently trained group count contributes to the validation oscillation, and whether concentrating the same additional budget on G4 is practically useful. Encoder, decoder, CE, physical transmission, energy constraints, RZF, and training objective are unchanged.

The motivating observation comes from existing seed20261107 training-validation logs: at steps9000/9250/9500/9750/10000, the last trained G was2/1/4/2/1, and the lowest validation NMSE belonged to that same G. This is evidence for a scheduling hypothesis, not proof of negative transfer or a statement about independent test performance. The shared backbone already received all10k warmup gradients; only the condition exposure and active prefix coordinates differ.

The batch does not start Stage C or train new encoder/decoder architectures. It does not repeat any original10k run. Test-role data remain sealed. All new results are validation results for deciding the next development step.

## Four fixed arms

Only seed `20261107` is used in this first screen. Every arm receives exactly `3024 = 28 * 108` new optimizer updates, including its first108-update technical pause. Total additional official updates are12096.

| Arm | Initialization | Training G schedule |
| --- | --- | --- |
| `blocked_universal` | Existing Hybrid10k | Original18 consecutive updates per G |
| `interleaved_universal` | Same Hybrid10k | Balanced interleaving with identical joint-condition counts |
| `specialist_g4` | Same Hybrid10k | Fixed G4 |
| `fdma_control` | Existing FDMA10k | Original FDMA layout |

The three Hybrid arms share exactly the same starting model state. FDMA starts from its corresponding separately trained10k model. All four start fresh AdamW and constant-LR scheduler states. Source optimizer history is preserved in the original files but is not carried into this new experiment. All codec parameters remain trainable; CE parameters and evaluation mode remain frozen.

Pinned source artifacts, relative to the03 repository:

| Source | Exact path | SHA256 |
| --- | --- | --- |
| FDMA | `runs/twc_v1_k8_codec_warmup/fdma/seed_20261107_jointv1/checkpoint_latest.pt` | `c0636e91fda1dd0118719f55665bce1841cfb72d9219354258174405c251b208` |
| Hybrid | `runs/twc_v1_k8_codec_warmup/hybrid/seed_20261107_jointv1/checkpoint_latest.pt` | `65e02758cab40908b85973a78012d9d0adb7e4ebe04a578348679ceab4ec79c0` |

Sources must be completed Stage-B artifacts with the expected arm, seed, step10000, full recipe and data protocol. Source paths are explicit and are never chosen by a latest-file search. CE remains `runs/twc_v1_k8_ce/seed_20261105/checkpoint_latest.pt` with its `calibration.csv`. Their fixed SHA256 values are respectively `0a5dc2e445e1d127252a6f0bfea488258a59e89fe767c9ee7dad99c9c488103f` and `d1a5dbbeb47220498e5fca25fa70e760d7b577eb9a0df7b88605f759f810d3b5`; mismatches stop the source check. The source check also constructs the existing replayable generator on CUDA without generating a training batch.

## Schedule, data and budget

The physical scenario is K8/Nt32/Nsc256 with256 feedback RE, pilot/key/shared data transmission, estimated UL CSI, and per-UE frame energy64. Hybrid uses the existing G1/G2/G4 geometry and branch normalization. FDMA retains its24 data coordinates per UE. The existing feedback noise convention and DL25 RZF path remain unchanged.

Training uses batch4, AdamW LR `2e-4`, weight decay `1e-5`, gradient clip1, and NMSE-only loss. No rate loss, auxiliary codeword loss, increased model capacity, or optimizer-budget advantage is introduced. New-stage local update `s` addresses the existing replayable training stream at cumulative update `10000+s`; training does not replay the first10k addresses. All arms use the same channel and standard-noise addresses at the same cumulative update.

Let `n = 10000 + s - 1`. The common public-condition schedule is:

```python
gamma = (0, 5, 10, 15, 20, 25)[n % 6]
Ep = (8, 16, 24)[(n // 6) % 3]
ratio = ((1, 3), (1, 1))[(n // 2) % 2]
G_blocked = (1, 2, 4)[(n // 18) % 3]
G_interleaved = (1, 2, 4)[(n + n // 18) % 3]
```

Both Universal schedules cover every one of the108 `(G,gamma,Ep,ratio)` conditions exactly once per108 updates and28 times during3024 updates. Their condition order differs, while marginal and joint counts match. A naive `G=(1,2,4)[n%3]` is invalid here because it locks G to the six-step gamma cycle. The configuration records the deterministic joint counts and the lightweight tests verify all28 complete cycles; runtime checkpoints separately verify completed optimizer updates.

The G4 specialist receives84 updates per `(gamma,Ep,ratio)` condition, compared with28 for the G4 slice of either Universal arm. Thus specialist-versus-Universal compares equal additional total updates with concentrated condition exposure. It cannot by itself establish that shared weights cause negative transfer. The primary scheduling comparison is `interleaved_universal - blocked_universal`.

## Continuation and failure behavior

All new outputs live under `runs/twc_v1_k8_schedule_screen_v1/`. Original warmup directories are read-only inputs. Fresh runs refuse existing output identities; interrupted runs require explicit resume. The controller performs source/environment checks, runs each arm's first108 updates, then automatically resumes that same3024-update plan. The first108 are not added again and are not a separate training experiment.

Checkpoint state distinguishes local update `0..3024` from cumulative update `10000..13024`. Resume loads tensors on CPU first, strictly checks model keys/shapes/dtypes, recipe/fingerprint, source identity and hashes, code identity, optimizer and scheduler states, and Python/NumPy/Torch CPU/all-CUDA RNG state before continuing. It does not reinitialize from the10k source on resume. Existing completed arms are verified and skipped. Failure retains evidence and stops; it does not change seeds, extend the budget, relax validation, or overwrite a source checkpoint.

The screen evaluates the predeclared final3024-update weights. It does not select a checkpoint from test results, choose a different final step per SNR, or compare a continued proposal against an uncontinued FDMA source. No step0 evaluation is required before training; the existing source diagnostics remain historical context.

Progress monitoring is deliberately small:16 validation samples, gamma10dB, Ep16, and key:shared ratio1:3. Both Universal arms monitor G1/G2/G4, the specialist monitors G4, and FDMA monitors its own layout. The monitor runs at the108-update technical pause, at756-update intervals and at an invocation's terminal boundary. These values show progress and condition-specific oscillation only; they never select a checkpoint or replace the final256-sample,90-condition comparison.

## Final paired validation

Evaluation uses256 complete multi-user validation-channel samples in64 batches of4. Validation-role batch addresses are1024 through1087, distinct from the warmup progress-validation addresses0 through3. The test role is never sampled. All methods, group counts and energy ratios receive the same actual H_dl/H_ul and standard pilot/data noise for each condition; content hashes establish pairing.

DL SNR is25dB. Feedback gamma is0/5/10/15/20/25dB, Ep=16, and both key:shared energy ratios1:3 and1:1 are reported separately. Both Universal arms are evaluated at G1/G2/G4, the specialist at G4, and FDMA at its single layout. This gives90 method/layout/gamma/ratio conditions:72 Universal,12 specialist and6 FDMA. FDMA is evaluated once per gamma and its preserved sample rows are used for the matching comparisons; it is not artificially duplicated to increase the sample count.

Main outputs include per-sample NMSE and sum rate, raw CSV.gz, condition summaries, signed paired contrasts, sample and batch95% intervals, frozen-state checks, input/source hashes and a manifest. Each sample represents one entire eight-user channel realization. Cross-SNR low0/5/10 and all-six averages are formed within a sample before sample/batch aggregation. The batch unit contains four complete channel samples; a user or subcarrier is not an additional independent sample. Intervals describe fixed-checkpoint validation sampling, not stability across training seeds.

Primary interpretation is whether balanced interleaving improves the Universal G4 slice over the blocked control without sacrificing the other G conditions. The specialist comparison addresses practical concentration of training. Comparisons with continued FDMA are reported separately. All six SNR points, both energy ratios, and negative differences remain visible. The single-seed screen cannot establish a final journal claim or guarantee that AS will outperform FDMA.

## Server workflow

Code preparation, lightweight checks and Git submission are performed on the Windows workspace. This batch adds six files: the independent schedule config, trainer, evaluator, controller, targeted test module and this document. The user runs the batch on the CUDA server after safely syncing `main` and checking the source files and GPU/process state. The server does not need Codex CLI.

```bash
python -B run_twc_v1_k8_schedule_screen.py --dry-run
python -B -u run_twc_v1_k8_schedule_screen.py --run
```

Read progress and final results without loading model weights:

```bash
python -B run_twc_v1_k8_schedule_screen.py --summary
```

Resume only after the former process has exited:

```bash
python -B -u run_twc_v1_k8_schedule_screen.py --run --resume
```

The controller summary identifies status/error, current stage, completed local and cumulative updates, exact result paths and paired results. Weights and raw samples remain in the independent server output directory. The user can return the summary text and the small result manifest/tables. Neither the manuscript nor historical result directories are updated automatically.

## Verification status

Executed locally: `python -B -S -m unittest tests.test_twc_v1_k8_schedule_screen -v`:18 passed,0 failures/errors/skips. These are standard-library scheduling, paired-statistics, metadata, CLI and subprocess-orchestration tests. They include the repeated-interruption regression that retains the verified108-step checkpoint hash before a later save. All five new Python files passed syntax checks. Help/dry-runs passed without Torch/NumPy/Sionna imports, checkpoint reads or experiment output creation. The two source model file hashes and CE/calibration hashes were read using CPU file tools.

These checks are not GPU acceptance. Real source deserialization, strict full-model loading, CUDA training/optimizer continuation, and the final256-sample validation have not been executed on the laptop. No complete CPU codec/CE forward/backward, CDL sampling, optimizer updates or SSH commands were run locally. Completion and performance claims require server receipts from this batch; interrupted/uninterrupted training bitwise equality remains unverified.
