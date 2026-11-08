# K8 BS-only decoder refinement screen

## Status and scope

Prepared on 2026-11-08 against the specified source implementation `ad3901743e1b18dc6829f91b1d43beca4f2c09ce`; the actual checkout and new experiment commit are recorded separately. This batch compares added Joint depth with a second, physical-residual-driven recovery round. It changes only the BS decoder. The UE, its condition embedding, CE, resource layout, transmit-energy/noise convention and physical DL CSI NMSE objective remain unchanged.

This is code preparation. No server source weight has been loaded locally, no source hash acceptance has been fabricated, and no real-checkpoint migration, CUDA training or performance result is claimed. Outputs are isolated under `runs/twc_v1_k8_decoder_screen_v1/`. Existing models, training/evaluation entries, paper, historical evidence and manual plotting files remain protected; 08 is not modified.

## Six arms

| Arm | Layout | Joint path |
| --- | --- | --- |
| `fdma_base2` | FDMA | Existing two Joint layers |
| `fdma_deep4` | FDMA | Existing two layers plus two added layers |
| `fdma_dc2` | FDMA | Existing two layers reused in a second residual-driven round |
| `g4_base2` | Hybrid G4 | Existing two Joint layers |
| `g4_deep4` | Hybrid G4 | Existing two layers plus two added layers |
| `g4_dc2` | Hybrid G4 | Existing two layers reused in a second residual-driven round |

All variants retain the spatial FC, spatial cross-attention and four Swin blocks. All BS parameters are trainable. The entire UE encoder, including its condition embedding and statistics, and CE remain frozen/eval; after a codec-wide `train()` call, the encoder must explicitly return to eval. Freezing the UE must not put the BS or complete chain inside no_grad/inference_mode during training.

## Fixed sources and server-side acceptance lock

Only these repository-relative sources are allowed:

```
runs/twc_v1_k8_rzf_matched_training_v1/training/fdma_nmse/checkpoint_latest.pt
runs/twc_v1_k8_rzf_matched_training_v1/training/g4_nmse/checkpoint_latest.pt
```

Expected source identity is seed 20261107, local step 12096, source-codec cumulative count 34192 and source implementation `ad39017`. These new source SHA256 values have not been provided. They are not filled using the older task-loss checkpoint hashes and are not guessed from a directory name.

Before any variant execution, the server must verify the original batch is completed and cross-check each actual checkpoint hash against its fixed manifest, `terminal_checkpoint.json`, batch-stage evidence, initialization record and complete training/progress logs. Architecture, full configuration/fingerprint and the historical provenance chain must agree. Missing evidence or a conflicting hash stops the batch: there is no checkpoint search, best-file substitution or old-manifest repair. Verified source identities are written once to a non-overwriting `source_acceptance.json`, then strictly checked on resume.

All three variants within a layout copy exactly the same accepted NMSE source state. Source checkpoint bytes and original evidence remain untouched and are not copied into the new output root. The new experiment HEAD is separate from the source implementation. Existing historical source and recovery checks are not relaxed.

## Deep4: append identity-initialized Joint layers

Keep the original two layers and every old parameter. Add two layers with the same specification. For each added layer, initialize the self-attention output projection/bias, cross-attention output projection/bias and final FFN linear/bias to zero. Other new parameters use normal initialization in an isolated RNG scope.

The zero residual outputs are intended to make the new layers act as identity at initialization. The static additional parameter count is 398080. This is an initialization hypothesis and structural count, not a declaration of successful complete-chain equivalence: the real source-to-base2/deep4 comparisons of H_hat and deployed RZF, actual complete model counts and peak memory must pass the existing accepted checks on the server. The controller must not enlarge tolerances or shrink the model to obtain acceptance.

## DC2: one second round driven by actual receive residuals

Round 1 is the original two-layer Joint receiver. It produces `context1` and the original head's corrected soft codeword `x1`. On actual active data REs, form

```
r_q = Y_data,q - sum_active(H_ul_est,k,q * x1_k,q)
d_q = solve(H_est^H H_est + ridge I, H_est^H r_q)
```

`x1` uses the scale of the physically normalized transmitted symbols. It is not renormalized. Only data REs enter; pilots are excluded. Use estimated UL CSI and the original receiver's N0/nominal-symbol-power/CE-error ridge approximation. No true H_ul, true codeword, true DL CSI or simulator-only error is an input. Recovered Hybrid keys are not treated as error-free pilots. The solve is a preconditioned residual feature, not a claim of an optimal Bayesian iteration or calibrated posterior; no explicit large matrix inverse is added.

Round 2 reconstructs queries from `x1`, reuses the original query projection, residual norm and the same two Joint-layer weights, and adds a new bias-free 2→128 projection of real/imaginary `d`. Existing reliability, CE-error, SNR and original observation memory remain available. Its new 128→2 output head has zero weight/bias and predicts `delta2`; `x2 = x1 + delta2`. There is no scalar gate or zero-head shortcut that bypasses the second round's computation graph.

Second-round self-attention permits same-user cross-coordinate information and other users that actually collide on the same RE. Hybrid G4 shared REs contain two active users; Hybrid key REs contain one. The first four FDMA packed coordinates are still FDMA data, not Hybrid keys. FDMA group id -1 must not merge all users into a common group. Inactive/padded tokens cannot enter effective recovery, and an active query cannot have every key masked.

Feed `x2` into the original FC. Spatial cross-attention still reads `context1`, not second-round context. Spatial FC, spatial cross-attention and Swin run only once. Keeping `context1` preserves the intended source behavior when `delta2` starts at zero; it does not dispense with real full-chain migration verification.

The static DC2 increment is 514 parameters: 256 in the bias-free residual projection, plus 256 weights and 2 biases in the second output head. A small-module local test instantiated this increment and confirmed 514; actual full-codec counts must still be verified on the server. No equal-parameter, equal-MAC or equal-latency claim is made, and the structure must not be altered merely to force that count.

## Strict migration, shared modules and gradients

Use independent receiver/refinement files; the existing `models/twc_v1_k8_joint_decoder.py` is unchanged. No hooks, global monkey patches or forward cache provide intermediate access. Stage codewords use explicit optional return fields and are reduced to small statistics before logging.

Preserve the shared instance relationship between `codec.decoder` and `joint_decoder.base_decoder`; do not construct a second independent spatial decoder. Migration checks all old keys, shapes, dtypes and values, admits only explicitly declared new keys, constructs the complete target state and loads with `strict=True`. Every old value must be copied unchanged. Normal checkpoint resume loads trained new parameters strictly and never reruns initialization.

The real batch4 preflight compares base2/deep4/DC2 H_hat and deployed RZF with the accepted source, checks the physical residual/layout/scale, trainable/frozen boundaries, power and active gradients, and records actual parameters, Joint execution counts and peak CUDA memory. At zero-output initialization, upstream new parameters can legitimately have zero gradient. Require a finite effective gradient on the new output head and finite other active gradients, with UE/CE gradients absent and their states unchanged; do not reject the path merely because every upstream new parameter is not nonzero yet. The first 108 updates also record whether new branches begin learning.

## BS-only training and common inputs

Each of six arms receives 6048 new BS optimizer updates, 36288 total. The first 108 are included and automatically continued after acceptance. Recipe: seed 20261107, physical batch4, fresh BS-only AdamW, LR 2e-4, weight decay 1e-5, constant scheduler, gradient norm clip 1. Only trainable BS parameters enter the optimizer; source optimizer state is not resumed.

Train on physical DL CSI NMSE only. There is no sum-rate term, codeword auxiliary loss or receive-residual fitting loss. RZF is used only for monitoring/evaluation. New training addresses are 34193–40240 with the existing gamma, Ep and two Hybrid energy-ratio schedule. Record source-codec count 34192, new BS updates and actual global data address separately. This does not mean the frozen UE receives another 6048 training updates.

For corresponding updates, the three variants of a layout share actual X, Y, channels and standard noise, verified by fingerprints. Across layouts, only channels and standard noise are required to match: different physical X/Y or receive powers are not an error. Keep all physical budgets and the source CE unchanged.

## Monitor, alpha selection and offline codeword diagnostics

Monitor at steps 1512, 3024, 4536 and 6048 on 64 complete multiuser validation samples, addresses 10240–10255, batch4. Record all six gammas at Ep16/DL25 and both G4 energy ratios separately. RZF alpha is fixed at **0.03 for both layouts**, unlike the preceding matched-training G4 monitor. Restore original modes/RNG after monitoring. No monitor-best checkpoint is selected.

Final evaluation compares six new final checkpoints and two frozen sources. First select one global alpha per checkpoint from the unchanged grid

```
[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
```

using 64 complete MU validation samples at 11264–11279. Average all six gamma conditions, with both G4 ratios equally weighted, then lock the selection before using the independent 256-sample validation at 12288–12351. Preserve fixed alpha0.03 and independently selected-alpha policies. Final results cannot reselect alpha, weights or SNR subsets, and test remains sealed. H_hat can be reused in memory within a batch for the RZF grid; no large persistent CSI cache is written.

Alongside CSI NMSE, sum/per-user rates, signal/interference and power checks, report the initial regularized/LMMSE soft-codeword error, first-round Joint error, DC2 second-round error and receive-residual energy. Codeword targets are the actual normalized transmitted symbols, with FDMA data and Hybrid key/shared reported separately. UE raw unnormalized outputs are not valid targets. True symbols are used only for offline diagnostics, never as decoder inputs or training loss. A lower receive residual without lower codeword error is not evidence of more accurate recovery.

Paired comparisons include deep4−base2, dc2−base2, dc2−deep4, G4−matching-FDMA for each variant, the corresponding gap interactions, and each new model relative to its frozen source. Planned final coverage is 144 fixed/selected conditions, 36864 raw rate sample rows, 1920 paired-summary rows and 184320 offline receiver-diagnostic rows; these are acceptance counts, not locally generated experiment results. Retain all SNRs, both ratios and negative outcomes. Compute across-SNR summaries within each complete MU sample, then sample/batch SE and 95% intervals; batch size is four. No user/subcarrier/SNR duplication inflates sample count. This is a single-seed composite structure screen: DC2 gains cannot be independently assigned to one of its added components.

## One server batch and compact handoff

The controller performs source acceptance, real batch4 migration/gradient/memory checks, six 108-update segments and automatic continuation, final alpha selection/validation and compact results. Keep only necessary rolling latest, terminal receipts, configuration and metrics, without copying source weights or saving large CSI caches. Strict explicit resume requires nonempty completed-stage evidence with matching hashes before skipping a stage; pointers must stay inside the owned output root. A verified 6048-step latest can recover only its small terminal/complete receipt after a final-write failure, with zero extra optimizer updates. Failure retains evidence and stops; no automatic model reduction, seed change or retraining from scratch is allowed.

The default summary is limited to about 40 lines and 8KB: status, six-arm progress, latest monitor means, complete six-point primary rate tables, key batch paired intervals and exact output paths. Full monitor, per-user metrics and detailed statistics require `--summary --details` rather than a default large JSON dump.

The following is for the existing server environment, not a command executed locally. It refuses synchronization while a same-user Python process is working inside this 03 checkout; it does not stop any process. The check precedes fetch/pull. Existing local modifications remain in place, and Git refuses an unsafe fast-forward rather than stashing or discarding them.

```bash
set -eu
cd -P ~/project/03_PRECODER
conda activate precoder
task_python="$(python -c 'import sys; print(sys.executable)')"
case "$task_python" in /*) ;; *) echo 'Absolute Python path required'; exit 1 ;; esac
for task_pid in $(pgrep -u "$(id -u)" -x 'python([0-9.]*)?' || true); do
  task_cwd="$(readlink -f "/proc/$task_pid/cwd" 2>/dev/null || true)"
  case "$task_cwd/" in "$PWD/"*) echo "03 Python process still active: $task_pid"; exit 1 ;; esac
done
test "$(git branch --show-current)" = main
test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git
git status --short --branch
test -z "$(git diff --cached --name-only)"
git fetch origin
test "$(git rev-list --count origin/main..HEAD)" -eq 0
git pull --ff-only origin main
git merge-base --is-ancestor ad3901743e1b18dc6829f91b1d43beca4f2c09ce HEAD
"$task_python" -B -S run_twc_v1_k8_decoder_screen.py --dry-run
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/decoder_screen_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_decoder_screen "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_decoder_screen.py --run > '$task_log' 2>&1"
printf 'Controller log: %s\n' "$task_log"
```

After completion or failure, activate the existing environment, resolve its absolute interpreter and return the complete compact summary:

```bash
conda activate precoder
cd ~/project/03_PRECODER
task_python="$(python -c 'import sys; print(sys.executable)')"
"$task_python" -B run_twc_v1_k8_decoder_screen.py --summary
```

Use `--summary --details` only when the detailed handoff is needed. After confirming the prior process has exited, explicitly resume with the same absolute interpreter:

```bash
"$task_python" -B -u run_twc_v1_k8_decoder_screen.py --run --resume
```

No separate per-arm smoke or continuation command is required.

## Actual local checks and remaining server work

Executed locally on 2026-11-08: `python -B -m unittest tests.test_twc_v1_k8_decoder_screen -v` passed 23 tests with 0 failures/errors/skips on CPU. Eleven checks exercise small real model/components without constructing the complete codec; twelve cover metadata, statistics, RNG and CLI/controller behavior. A separate `python -B -S` test run passed 11 checks and explicitly skipped 12 because real site-package dependencies were unavailable in that mode. All seven Python files passed Python 3.10-compatible AST and compile checks. All three entry points passed standard-library dry-runs; standard-library `--summary` correctly reported not_started without loading weights or creating the output root.

A Git diff confirmed all 37 old `SOURCE_FILES` were unchanged. The five protected pre-existing plotting/figure files retained their recorded SHA256 values, and `runs/twc_v1_k8_decoder_screen_v1/` was not created locally.

These checks cover component identity initialization and strict source-copy/alias rules, complex residual axes/scales and activity masks, zero-head gradient behavior, BS-only metadata/update counts and source-lock changes, signed paired statistics, compact-summary size/core-comparison coverage, nonempty completed-stage hash verification and the controller's included108/explicit-continuation call path. The monitor check exercises mode/buffer/default-RNG restoration with a small real module and explicitly synthetic data/forward work, not a real CUDA/64MU channel run. The implemented failure paths additionally preserve corrupted alpha/small-JSON bytes while reporting error/unknown summary fields and support zero-extra-update terminal-receipt recovery; actual server execution remains pending. Key structure comparisons have explicit summary priority instead of being lost to a sort-order cutoff.

No complete model is built locally; no real source checkpoint loading, CDL generation, training, GPU evaluation or SSH connection is performed. CPU component identity or gradient tests are not real-checkpoint full-chain migration or CUDA/RNG/memory acceptance. No large environment is installed and missing dependencies are not fabricated.

Server work remains for actual source SHA cross-lock/completion/provenance acceptance, strict full-chain migration, physical residual/scale and masks, real CUDA gradients/frozen-state checks, actual parameter and peak-memory counts, six complete BS training runs and all monitor/evaluation results. Static identity initialization and the 514-parameter formula are not labeled verified performance or real-checkpoint acceptance. The existing source weights and all historical evidence are preserved.
