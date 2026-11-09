# K8 physical-symbol / semantic-latent split screen

## Current status and evidence boundary

Prepared and lightly verified on 2026-11-09 from the specified decoder-screen implementation `b6336de4c80588e8d1ddb57a0607f774182a0f67`. The user reports accurate initial soft symbols but large subsequent Joint-to-transmit distances: six-gamma linear NMSE approximately 0.002656→7.866 for FDMA, 0.002417→4.826 for Hybrid key, and 0.018817→0.771 for Hybrid shared. The supplied audit did not identify a clear slot or energy-normalization mismatch. These observations were provided by the user, not rerun locally.

The old Joint intermediate was optimized for final CSI reconstruction, so it need not remain a physical symbol estimate. This batch preserves that semantic path and tests a separately supervised physical estimate behind an explicit detached interface. It does not assume Hybrid will overtake FDMA, use receive-residual fitting as a loss, or force the old semantic latent to equal transmitted symbols.

The namespace is `runs/twc_v1_k8_physical_split_v1/`. Source SHA acceptance, full-chain migration, CUDA checks, training and evaluation remain server work. The paper, historical results, original models/entries and 08 are unchanged.

## Cleanup completed before implementation

The standard-library `cleanup_twc_v1_k8.py` defaults to preview; deletion requires `--apply`. `--report` may name a new file directly inside this checkout's `cleanup_reports/`, and existing reports are never overwritten. It scans only immediate root bytecode/tool caches and source directories `configs/`, `models/`, `tests/`, `utils/`, `paper/scripts/`. Git-tracked files, links/reparse points, environments, outputs/data/archives, locks and unknown/manual files are excluded. Bytecode must have a matching existing source file; supported tool-cache records must match their known cache form.

Active repository Python processes, uncertain Windows cwd, open files or inspection failure cause conservative skips. A server controller invoking cleanup in a child process may therefore receive a valid zero-deletion/skip report; it must not claim server caches were removed. Hard-linked file data are counted once, and recoverable allocation is not claimed while a retained link remains. Logical bytes and the observed disk-free change are separate fields; unknown Windows allocation remains unknown.

Local preview and apply were actually executed before the other physical-split implementation work. Apply receipt:

```
cleanup_reports/cleanup_20261109T095812_987567Z_16df4a45.json
```

Only one 24819-byte bytecode file, produced by this round's cleanup-script syntax validation, was deleted. No pre-existing bytecode file was found. Five empty cache directories were removed: root `__pycache__`, and those under `configs`, `models`, `tests`, `paper/scripts`. Measured D-drive free space was 465239392256 bytes before and after deletion: observed delta 0; exact attributable filesystem allocation was unavailable. This is not reported as a substantial disk recovery.

The five protected manual plotting/figure files and root `.gitignore` retained their SHA256 values. Tracked Git status and tracked-deletion state were unchanged; new cleanup code and non-Git receipts are intentional additions. No weight, raw sample, archive, historical log, lock or source evidence was deleted. Metadata-only retained candidates were:

| Local path | Logical bytes | Decision |
| --- | ---: | --- |
| `runs/twc_v1_k8_codec_warmup/` | 873122688 | Retain source/history; no proof of a redundant disposable copy |
| `runs/twc_v1_k8_ce/` | 3640629 | Retain CE source/calibration |

Other later source-chain outputs were not present locally; their server contents are not claimed cleaned. Large weights were not read/hashed merely to inventory sizes. Receipts stay uncommitted.

## Four arms and fixed sources

| Arm | Layout | Physical head | Reconstruction bridge |
| --- | --- | --- | --- |
| `fdma_control` | FDMA | Supervised | Stored but disabled and excluded from optimizer |
| `fdma_split` | FDMA | Same supervised head | Enabled/trainable |
| `g4_control` | Hybrid G4 | Supervised | Stored but disabled and excluded from optimizer |
| `g4_split` | Hybrid G4 | Same supervised head | Enabled/trainable |

Both arms of each layout start from the corresponding completed Base2 state at these fixed paths:

```
runs/twc_v1_k8_decoder_screen_v1/training/fdma_base2/checkpoint_latest.pt
runs/twc_v1_k8_decoder_screen_v1/training/g4_base2/checkpoint_latest.pt
```

The server cross-checks original completed batch, manifest, terminal receipt, actual checkpoint SHA/state and the complete configuration/provenance chain, then writes an immutable source lock. SHA values are not guessed, copied from earlier checkpoints or discovered by searching for a replacement. The historical source implementation is b6336de; the new HEAD is a separate identity. Current STATUS updates must not be substituted into historical source metadata: the archived old STATUS hash remains available in Git, while unchanged historical computation-source hashes stay exact.

Strict migration copies every old key/value/shape/dtype and preserves the shared instance of `codec.decoder` and `joint_decoder.base_decoder`. Only the declared new physical-head/bridge parameters are admitted. Old source weights/metadata stay read-only. Normal resume preserves trained new parameters rather than migrating again or zeroing them.

## Two paths and intentional gradient boundaries

The original reconstruction path remains:

```
context1 -> original residual head -> z_sem
        -> z_recon -> original FC / spatial attention(context1) / four Swin blocks -> H_hat
```

`z_sem` is a reconstruction latent, not a guaranteed physical codeword estimate. The new physical head is a shared token MLP, 128→64→2 with GELU and only its final linear weight/bias zero initialized:

```
x_phy = soft.detach() + physical_head(context1.detach())
```

Mask inactive slots; do not renormalize `x_phy`. At initialization it equals the original soft estimate. From only `x_phy`, actual received data REs and H_ul_est, reconstruct the receive residual and use the already accepted regularized matched-residual solve to obtain `d_phy`. Pilots are excluded; active colliding users are summed using the true layout mapping. Estimated UL CSI and the existing ridge approximation are used, not true UL CSI or error-free recovered key pilots. `z_sem` is never substituted into this physical residual.

The bridge is 5→64→2 with GELU, again with only the last linear zero initialized. Its five features are real/imaginary `x_phy−soft`, real/imaginary `d_phy`, and the existing reliability scalar. The complete feature vector is detached before the bridge:

```
z_recon = z_sem + physical_bridge(detach(physical_features))
```

The residual solve is a feature/diagnostic computation under no_grad, not an additional trainable objective. Spatial attention still uses original `context1`, and FC/spatial attention/Swin run once. Both variants store both new modules: the small-component test confirms head count8386, bridge count514, total addition8900. Full-codec counts remain server acceptance. Control never executes the bridge and excludes it from both optimizers; no equal-MAC or equal-latency claim is made.

| Loss | Permitted parameter updates | Explicitly blocked |
| --- | --- | --- |
| Physical codeword NMSE | `physical_head` only | Original Joint/BS, bridge, UE/condition embedding, CE |
| Physical DL CSI NMSE | Original BS and active bridge | Physical head, UE/condition embedding, CE |

Codeword targets are detached **actual energy-normalized transmit symbols**: FDMA `data_symbols_tx`, Hybrid `key_symbols_tx` and `shared_symbols_tx`. They are accessed after the deployable forward only for training supervision/offline diagnostics; they are not decoder inputs. FDMA loss is the active-data codeword NMSE. Hybrid loss is `0.5*(key branch NMSE + shared branch NMSE)`, each normalized within its branch, first per complete MU sample and then averaged across the physical batch. It is not weighted by token count or branch energy. No beta/lambda calibration is required because the disjoint losses use separate optimizers.

## Data iterations and two optimizer clocks

All arms use seed20261107, physical batch4, AdamW LR2e-4/weight decay1e-5, constant schedulers and gradient norm clip1 separately for the two parameter groups. Source optimizer state is not reused. The UE, condition embedding and CE remain frozen/eval; code restoring codec.train must restore their eval state.

| Iterations | Head optimizer | Original-BS/active-bridge optimizer |
| --- | --- | --- |
| 1–108 | Update every iteration | Frozen; 0 updates |
| 109–6048 | Update every iteration | Update every iteration; 5940 updates |
| Total per arm | 6048 | 5940 |

Four arms therefore use 24192 new data iterations and head updates, 23760 new BS optimizer updates, UE/CE zero updates. The first108 technical segment is inside the6048 budget and automatically continues without resetting either scheduler. Checkpoints/resume record separate optimizers, scheduler epochs, RNG, phase, source counts and data address. They reject a head/BS step mix-up or fallback to fresh initialization.

The source has codec count34192, prior BS-only count6048 and last data address40240. New training uses `codec_train`40241–46288, preserving six gamma points, original Ep/key/shared schedule, G4 and the two energy ratios, 256 RE and energy64 per UE. These data addresses are not new UE training updates. Same-layout control/split share actual X/Y/channels/standard noise and retain fingerprints; cross-layout pairing requires common channels/standard noise, not equal X/Y or receive powers.

## Monitoring and final comparison

| Purpose | Role / addresses | Complete MU samples |
| --- | --- | ---: |
| Monitor at1512/3024/4536/6048 | `validation`14336–14351 | 64 |
| Final-only alpha selection | `validation`15360–15375 | 64 |
| Independent final validation | `validation`16384–16447 | 256 |

Monitor restores modes/RNG and reports six gammas, Ep16/DL25 and both G4 ratios using fixed RZF alpha0.03 for both layouts. Only the prescribed last6048 weights are used, never monitor-best.

Final evaluation includes four new methods and four unchanged old controls: `old_fdma_base2`, `old_g4_base2`, `old_fdma_deep4`, `old_g4_dc2`. Old controls are source-locked read-only and are not retrained. Keep fixed-alpha0.03 as the primary report. The independent final-only alpha diagnostic uses `[0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0]`, one global alpha/checkpoint over all six gammas and both G4 ratios equally weighted. Lock selection before final validation, retain both policies and do not rewrite training/monitor history or select on final results. Test remains sealed.

Report all conditions and negative outcomes, including DL CSI NMSE, MU/per-user rates, signal/interference, normalized energy/RZF checks, physical soft/head codeword errors and receive residuals. `z_sem`/`z_recon` distances to tx are explicitly **latent diagnostics**, not physical estimation accuracy. NMSE is labeled linear or dB; aggregate linear errors before converting to dB. Receive-residual output is mean unnormalized power and cannot be compared directly across layouts as an accuracy ranking. A lower residual alone is insufficient evidence of a better physical estimate.

Key comparisons are split−control within each layout, matched Hybrid−FDMA, bridge/layout gap interactions and changes relative to the read-only old strong controls. Across-SNR means are computed inside the same complete MU sample before sample/batch paired SE/95% intervals; batch size4, no inflated user/subcarrier/SNR sample count. Intervals are single-seed fixed-checkpoint validation sampling, not training-seed stability. Optional bridge-off/physical-from-soft switches serve server preflight interface checks; this version does not add intervention rows to the unified performance evaluation or use them for selection.

## Controller and server boundary

The controller performs bounded cleanup, immutable source acceptance, real batch4 migration/gradient/interface/memory checks, four108-to6048 runs, final selection/validation and compact packaging. It checks zero-bridge equivalence to the source using unchanged accepted H_hat/RZF tolerances, strict aliases/old values, actual normalized targets/residual mapping and both gradient boundaries. A deliberately enabled test bridge must let legal physical-input changes affect H_hat. Zero initialization can make upstream new gradients temporarily zero; the bridge output layer must still have an effective gradient. No failed performance threshold automatically changes seed, length or recipe.

Strict explicit resume verifies both optimizer clocks, state and receipts and skips only hash-verified completed stages. It neither overwrites existing reports nor unlocks/terminates another process. Keep necessary rolling latest/terminal receipts and metrics, without copying source checkpoints or large CSI caches. Default summary stays around40 lines/8KB with progress counts, complete primary six-point curves, key batch CI, physical-error changes, exact package path and an error/log tail on failure; detailed output requires explicit details.

The existing server `precoder` environment and an absolute Python interpreter are used. Process checks precede fetch/pull and no running server code is updated. This block is delivered for SSH execution; it is not run on the lightweight client. It stops before synchronization if another same-user Python process is working in this 03 checkout and does not kill/unlock anything.

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
git merge-base --is-ancestor b6336de4c80588e8d1ddb57a0607f774182a0f67 HEAD
"$task_python" -B -S run_twc_v1_k8_physical_split.py --dry-run
mkdir -p terminal_logs
task_log="$PWD/terminal_logs/physical_split_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
tmux new-session -d -s k8_physical_split "cd '$PWD' && '$task_python' -B -u run_twc_v1_k8_physical_split.py --run > '$task_log' 2>&1"
printf 'Controller log: %s\n' "$task_log"
```

For the compact return output:

```bash
conda activate precoder
cd ~/project/03_PRECODER
task_python="$(python -c 'import sys; print(sys.executable)')"
"$task_python" -B run_twc_v1_k8_physical_split.py --summary
```

Use `--summary --details` for full records. After confirming the preceding process has exited, explicitly resume without resetting the two clocks:

```bash
"$task_python" -B -u run_twc_v1_k8_physical_split.py --run --resume
```

The controller applies the accepted startup-only low-load GPU check; it does not promise continuous automatic pausing. No SSH action is performed locally.

## Actual local checks and pending server validation

The final combined suite ran33 tests:32 passed,0 failures/errors and1 skip for real Windows symlink creation, which required unavailable privilege1314. It comprises10 physical-component tests,12 controller/statistical/training tests and11 cache-safety tests. A separate `-B -S` run executed the same33 cases with20 passes and13 explicit skips:12 required the real Torch/NumPy site packages and one required the unavailable Windows symlink privilege. All nine Python files passed Python3.10-compatible AST/compile checks. Three entry points passed standard-library dry-runs, and the no-start summary returned normally without creating a training output root.

Numerical component checks confirm actual MLP counts and the two disjoint loss paths. The dual-optimizer test uses real `torch.save`/`torch.load` through BytesIO and the trainer's actual `validate_runtime_resume`, verifying head108/BS0 to next head109/BS1 and preservation of trained parameters. The monitor test checks real default RNG restoration on normal/exceptional exits using a small real module and explicit synthetic channel/forward fixtures; it is not a real CDL/64MU CUDA monitor. The full-result summary regression preserves the final failure-log lines and exact paths within both line and byte budgets. Cache tests verify real hard-link accounting, tracked/evidence preservation, preview/report refusal, active/uncertain process and held-file skips, generated-cache signatures and a failed protection check's non-overwriting receipt before raising.

The final protection audit found all45 prior source/document/test files outside the explicitly updated STATUS page unchanged. All six protected file hashes (the five plotting assets and .gitignore) and the original tracked-deletion set were preserved. Cleanup receipts remain local and outside the staged allowlist; the new experiment output root was not created.

No complete model, source checkpoint, CDL, training or GPU evaluation is run locally. Real source acceptance, actual full-codec counts, zero-bridge full-chain replay, physical/interface/gradient CUDA checks, peak memory, full dual-optimizer pause/resume and all performance results remain server validation. The prior decoder observations are user-provided evidence, not a local performance evaluation.
