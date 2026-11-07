# K8 ordered-window encoder controlled screen: v2

## Current v2 and preserved v1 failure

On 2026-11-07 the user returned a real server v1 migration-preflight failure. Reported raw-code maximum errors were about `1.34e-7` to `5e-7`, normalized-transmit error `8.28054e-7`, H_hat error `1.32419e-6`, and RZF error `2.54094e-5`. The `ordered_fdma`, gamma 0 comparison failed the original RZF tolerance. DL NMSE and rate were not accepted by that receipt, and none of the four training arms started. These are user-provided server observations; this documentation update does not claim the server report was read or rerun locally.

The v1 implementation in commit `795a0cc` and its output directory `runs/twc_v1_k8_ordered_encoder_screen_v1/` remain historical evidence. No old checkpoint, manifest or failure receipt is rewritten or removed. Current execution uses the independent namespace `twc_v1_k8_ordered_encoder_screen_v2` and architecture `twc_v1_k8_ordered_window_encoder_v2`. It preserves the original global-mean reduction and adds a zero-initialized ordered-window difference readout. The fixed tolerance table and RZF calculation remain unchanged. The v2 CUDA migration/gradient checks and four training runs have not yet been executed on the server.

## Question and evidence boundary

This batch asks whether retaining the order of the existing Swin windows improves reconstruction and downlink task performance relative to the current global-mean encoder. It does not assume that pooling alone explains the Hybrid–FDMA difference. The v2 encoder retains both original 128-wide heads and adds only two bias-free ordered-window residual weights. The original encoder remains available, and the BS Joint receiver, physical link, frozen CE, branch-energy normalization and RZF are retained.

The user reports that the preceding schedule screen completed four additional3024-update runs, with cumulative update13024. At key:shared energy ratio1:3, the six-SNR mean rates were10.70676 for blocked Universal G4,13.47553 for interleaved Universal G4,15.70854 for the G4 specialist and19.96734bps/Hz for continued FDMA. These are user-provided validation results, not locally rerun evidence or independent test results. They motivate retaining the same improved encoder opportunity for FDMA rather than comparing a modified Hybrid against an unmodified FDMA.

## Read-only reference and exact adaptation

The ICC reference was inspected read-only in the neighboring08 project at commit `bfa04279dd55873a61ecdd476c0144a8025cd496`. No08 code, weights or result files are changed or imported at runtime. The relevant reference files are:

| Reference | Observed source SHA256 / role |
| --- | --- |
| `src/icc08/resource_research/round6_resource/codec.py` | `843b400676699d194f456461fe468a535c47ea8f929ac33241cb6e41606ab8d1`;4096-wide ordered-window compression head |
| `src/icc08/shared/swin_codec.py` | `08249826ea8a01832a6ebcfda8aa1e0d6358b2863500f64be3258aa1f3875d78`; shared non-shifted window blocks and patch geometry |
| `src/icc08/resource_research/round3_spatial/codec.py` | `a933df5b2e2e0fb868f7fc26036dc691f6bdd0bfd4b7f527c2fecb934b8e0031`; defines the pooling imported by the Round6 reference |

The Round6 reference has a K4/216-complex-codeword interface. This experiment adapts the ordered pooling to03's existing K8 key4/shared160 maximum interface, G4 active prefix40, FDMA24, and its current pilot/data protocol. It is an ICC-style ordered-window variant, not an unchanged reproduction of the ICC model or communication system.

## Current v2 readout and parameter count

Let `t` be the output of the original LayerNorm applied once to the existing 512×128 Swin tokens. The original global feature is computed with the same operation as the mean encoder:

```python
g = t.mean(dim=1)  # the original 512-token reduction, not mean(window_means)
p = ordered_window_pool(t)  # 32 ordered 128-vectors from the same normalized tokens
d = concat(p[0]-p[31], ..., p[30]-p[31])  # 3968 features
y = W0(g+c_G) + b + Delta(d)
```

The norm and Swin backbone are each evaluated once. Both original `W0` and `b` remain trainable and are copied unchanged from the source. The two new `Delta` projections have no bias and start at exact zero; their forward paths still execute at zero initialization so gradients can reach their weights. The G embedding is added through the preserved original base path. It is not added to a second global feature reconstructed from window means. The residual receives window differences, so any common G embedding would cancel there.

| State key | Shape in v2 | Initialization |
| --- | --- | --- |
| `encoder.key_head.weight` | `[8,128]` | Original source value, unchanged |
| `encoder.shared_head.weight` | `[320,128]` | Original source value, unchanged |
| `encoder.key_window_residual.weight` | `[8,3968]` | Zero, no bias |
| `encoder.shared_window_residual.weight` | `[320,3968]` | Zero, no bias |

Every original state key, shape, dtype and value is retained. Migration admits only the two listed new zero weights, assembles the exact target key set and uses `strict=True`. Normal v2 checkpoint restoration loads the trained residual weights strictly and never reinitializes or migrates them again. Source files remain read-only.

The added parameter count is `31×128×(8+320) = 1,301,504`, entirely on the UE side; BS count is unchanged. The retained base weights plus residual weights have the same stored readout count as a general 4096-wide v1 head. Actual complete model/UE/BS counts and active-path finite nonzero residual-gradient checks remain server acceptance items. No equal-parameter, equal-MAC or equal-latency claim is made.

### Ordered-head representation and G-condition constraint

For an arbitrary ordered linear head with 32 window blocks `A_j`,

```
y = sum_j A_j(p_j+c_G) + b,
W0 = sum_j A_j,
Delta_j = A_j - W0/32,  j=1..31.
```

Since `g = sum_j p_j/32` in exact arithmetic,

```
W0(g+c_G) + sum_(j=1..31) Delta_j(p_j-p_32) + b
= sum_j A_j p_j + (sum_j A_j)c_G + b.
```

Conversely, `A_j = W0/32 + Delta_j` for the first 31 windows and `A_32 = W0/32 - sum_(j=1..31)Delta_j`. Thus v2 retains the full linear ordered-head representation with the same common-G condition constraint. Its zero-residual initialization reuses the original numerical global reduction rather than relying on floating-point equality between two reduction orders. This representation identity does not imply equal gradients, AdamW function updates or optimization trajectories across parameterizations.

## Historical v1 readout and migration

The following describes the superseded v1 parameterization and the reasoning behind its initialization. It is preserved to explain the failed server preflight, not as the current v2 implementation.

The original encoder applies LayerNorm to512 tokens of width128 on a64×8 delay–angle patch grid and takes their global mean. The ordered variant instead partitions that same grid into4×4 token windows, averages within each window and retains the16×2 window grid in row-major order: delay-window index first, then angle-window index. The32 ordered128-vectors are flattened to4096 features and passed to the existing key and shared output heads.

The AD transform, patch4, four non-shifted Swin blocks, width128, heads4, LayerNorm, Re/Im output ordering, G embedding, prefix selection and feedback dimensions remain unchanged. No position embedding, new attention block, auxiliary loss or extra neural path is added. The same G embedding is added to each128-dimensional window vector before flattening.

For the default geometry, only these weight shapes change:

| State key | Mean encoder | Ordered encoder |
| --- | --- | --- |
| `encoder.key_head.weight` | `[8,128]` | `[8,4096]` |
| `encoder.shared_head.weight` | `[320,128]` | `[320,4096]` |

The bias shapes remain8 and320. The structural parameter increment is `(4096−128)×(8+320) = 1,301,504`, in both UE and full-model storage; BS parameter count is unchanged. This is a formula cross-check, not a substitute for server instantiation: the CUDA preflight records actual old/new total, encoder and decoder counts. The experiment does not claim equal parameters, MACs or latency, and it does not measure runtime speedup.

### Historical v1 head expansion

Let normalized token vectors be `t[w,i]`, with32 windows and16 tokens per window. Let `p[w] = mean_i(t[w,i])` and `c_G` be the original G embedding. Since all windows have the same size,

```
global_mean(t) + c_G = (1/32) * sum_w (p[w] + c_G).
```

For either original head `y = W_old(global_mean(t)+c_G)+b_old`, initialize

```
W_new = concat([W_old/32] * 32, axis=input_dimension)
b_new = b_old.
```

Applying that head to `flatten(p[w]+c_G)` yields the same expression in exact arithmetic. Copying an unscaled matrix, adding G after flattening with an incompatible shape, or dividing the bias by32 would not preserve this function. Independent window weight blocks start equal but remain independent trainable parameters; future learning can distinguish window positions.

V1 migration checked the complete source and target key sets, tensor shapes and dtypes, allowing only the two head-weight shape expansions. All other parameters and persistent buffers were copied exactly and loaded with `strict=True`. This expanded-head initialization is not used by v2. Its mathematical equivalence did not prevent the reported downstream RZF tolerance failure.

Floating-point operation order differs from the global mean, so exact mathematical equality does not promise bitwise equality. Tolerances are fixed before any server run:

| Compared quantity | rtol | atol |
| --- | ---: | ---: |
| Raw and normalized transmitted codewords | 1e-5 | 1e-6 |
| H_hat and deployed RZF | 1e-4 | 1e-5 |
| DL NMSE | 1e-4 | 1e-6 |
| Downlink rate | 1e-4 | 1e-4 |

The independent maximum energy and per-subcarrier RZF-power errors must remain below1e-4 and1e-5 respectively. Small CPU tensor algebra checks and server comparisons report actual maximum absolute/relative errors. The server uses paired inputs/noise for the original and migrated paths and records backend/precision settings. Failure retains evidence and stops; the controller does not enlarge tolerances to pass.

The historical v1 small CPU algebra check observed FP32 maximum absolute error `4.76837158203125e-7` and maximum relative error `4.576317564897181e-7`; FP64 maximum absolute error was about `8.88178e-16`. Those checks established the head algebra, not complete-model or real-checkpoint migration acceptance. The returned server failure is retained above. In the executed v2 small-tensor checks, zero residuals preserved both original heads bitwise in FP32 and FP64: `torch.equal=True`, maximum absolute and relative error 0. The FP64 arbitrary ordered-head representation check had maximum absolute error `1.7763568394002505e-14`. These algebra results do not establish real-checkpoint CUDA equivalence.

For v2, the server preflight collects every comparison field for both sources and every prescribed condition, including NMSE and rate, even when an earlier field fails. It also runs the original RZF calculation twice on the same H_hat and records a separate numerical replay comparison, `torch.equal` result, backend settings and measured errors. This repeatability check requires exactly equal repeated RZF tensors and unchanged H_hat; any difference is separately reported as `RZF_repeat` and stops acceptance. It also records the comparison under the original RZF tolerance. This same-input, same-operator replay check is separate from the original-versus-v2 migration check, whose RZF tolerance remains unchanged. Any necessary failure rejects the preflight; neither RZF nor the cross-model tolerances are changed to obtain acceptance.

## Four fixed training arms and source provenance

All arms use seed20261107, batch4 and exactly3024 new optimizer updates, including the first108-update technical pause. The total is12096 new updates. Both Hybrid arms use fixedG4: four groups of two UEs. No G1/G2 training is added.

| New arm | Source family | Encoder |
| --- | --- | --- |
| `mean_fdma_control` | completed schedule-screen FDMA | original global mean |
| `ordered_fdma` | same completed FDMA | migrated ordered-window |
| `mean_g4_control` | completed schedule-screen G4 specialist | original global mean |
| `ordered_g4` | same completed G4 specialist | migrated ordered-window |

Exact source files, relative to03:

| Source | Path | SHA256 |
| --- | --- | --- |
| FDMA | `runs/twc_v1_k8_schedule_screen_v1/training/fdma_control/checkpoint_latest.pt` | `d0c3537d4e22bd061b387251f22030961829fc231126d1126b0cefde9b8f7d39` |
| G4 | `runs/twc_v1_k8_schedule_screen_v1/training/specialist_g4/checkpoint_latest.pt` | `ed92ca1c6688ee24f5515d3d281dc012bef27a04e0899ae19f55bf97335d20e8` |

The sources must identify the expected original arm, seed20261107, local step3024 and cumulative update13024, with the original complete recipe, model state, optimizer/scheduler/RNG and completed source evidence. Merely finding a file is insufficient. The accepted source implementation is commit `8096ed965027a92f6e0738c7653b4c3e75f71b8c`; that provenance is checked against its archived Git/source identities, separately from the new experiment's commit and source hashes. A later new HEAD is expected and does not mean the old source was modified. Historical manifests are never rewritten to the new hash values.

Each new arm starts a fresh AdamW and constant scheduler, using LR2e-4, weight decay1e-5 and gradient clip1. All codec parameters remain trainable; CE stays frozen/eval. Loss remains mean per-user DL NMSE with no rate or auxiliary loss. Original optimizer states are evidence for the source experiment, not the new optimizer initialization.

Function preservation applies at initialization. The original base weights and 31 independent window-difference blocks give AdamW a different parameterization from the mean encoder, so identical optimizer recipes do not imply identical subsequent function updates or optimization trajectories. This is retained as part of the prescribed ordered-readout experiment and does not justify changing the agreed learning rate or claiming an equal-parameter comparison.

New local step1..3024 maps to cumulative training address13025..16048. The existing gamma, pilot-energy and two key/shared-ratio schedules continue from the cumulative phase; corresponding steps share the same actual channels and standard pilot/data noise across arms. Training does not replay addresses1..13024. New output identity and checkpoint metadata distinguish local step, cumulative step, architecture, source experiment commit and new experiment commit.

## CUDA batch execution and recovery

The single controller runs source checks, the necessary migration/CUDA acceptance, each arm's108-update pause and automatic continuation to3024, common validation, and a small result package. Current outputs are isolated under `runs/twc_v1_k8_ordered_encoder_screen_v2/`; no v1 receipt is reused as a passed v2 stage. Fresh execution refuses existing run identities. Explicit resume validates code, configuration, source/CE identities, complete model/optimizer/scheduler/RNG, local/cumulative update counts and completed-stage evidence before continuing. CPU RNG byte tensors remain on CPU during loading; models and optimizer states then move through their normal device-loading interfaces. A completed stage is reused only after its evidence is verified, and original source weights are not overwritten.

Startup checks inspect the selected GPU and other tasks using the accepted low-load shared-GPU policy: GPU utilization at most20% and at least4096MiB free memory. Other processes may remain present when those conditions hold; no colleague process is stopped. The controller records startup observations and has no promise of continuous GPU-load monitoring or automatic pause when another job appears. Server source files must remain unchanged during the run. Any necessary identity, numerical or migration check failure retains evidence and terminates with a nonzero exit.

The batch uses explicit `--run`, standard-library `--dry-run`, `--summary`, and `--run --resume`. It does not require SSH Codex. Actual CUDA/model acceptance has not been executed on the laptop.

## Paired validation and interpretation

Only the predeclared final3024-update weights are evaluated; there is no result-dependent best selection. The final data are256 complete K8 MU validation samples in64 batches of4, using fresh validation-role addresses2048..2111. The data protocol and addresses are checked against the original replayable generator; the test role remains sealed.

All six gamma points0/5/10/15/20/25 are retained, with Ep16 and DL25. G4 is evaluated separately at energy ratios1:3 and1:1. Each FDMA arm is evaluated only once per gamma, giving36 total conditions:12 FDMA and24 Hybrid. Actual H_dl/H_ul and pilot/data standard-noise fingerprints establish pairing across arms and energy conditions.

The predeclared primary contrasts are:

1. `ordered_g4 − mean_g4_control`.
2. `ordered_fdma − mean_fdma_control`.
3. `ordered_g4 − ordered_fdma`.
4. `(ordered_g4 − ordered_fdma) − (mean_g4_control − mean_fdma_control)`.

The interaction is computed from the same sample's four rates, not by subtracting independent means or CI endpoints. For each contrast and metric, report every gamma, the low0/5/10 average and the full-six average. Cross-SNR averaging takes place within the complete MU channel first. Then compute sample SE and batch SE using sample means and `sample_index//4` batch means respectively, with ddof1 and CI=`mean±1.96SE`. Negative differences remain signed. A subcarrier or UE is not an extra independent sample. Ratios are reported separately; FDMA rows are reused for pairing without duplicating observations.

Outputs include DL NMSE, sum rate, desired signal/interference power, energy and RZF-power checks, actual parameter counts, raw samples, paired summaries, input/code/source manifests, and a small package that excludes weights and compressed raw samples. Hybrid improvement alone is insufficient: the modified FDMA remains the principal matched comparator. Single-seed validation intervals describe sampling conditional on the fixed checkpoints, not training-seed stability or final journal performance.

## Server entry points and verification status

```bash
python -B run_twc_v1_k8_ordered_encoder_screen.py --dry-run
python -B -u run_twc_v1_k8_ordered_encoder_screen.py --run
python -B run_twc_v1_k8_ordered_encoder_screen.py --summary
# Only after the previous process has exited:
python -B -u run_twc_v1_k8_ordered_encoder_screen.py --run --resume
```

The delivery includes the full safe-sync/environment/startup command block. Return the complete `--summary` output after completion or failure; it identifies exact source/checkpoint evidence, migration errors, parameter counts, training progress, signed contrasts and result-package paths.

Historical v1 local verification ran20 targeted tests with20 passes and no failures/errors/skips. Those local checks did not establish full-checkpoint CUDA equivalence, as demonstrated by the subsequently returned v1 RZF failure. The archived8096 source identity remains checked exactly on the server; it is not normalized to the current new HEAD.

Current v2 local verification: `python -B -m unittest tests.test_twc_v1_k8_ordered_encoder_screen -v` ran 29 targeted tests with 29 passes and 0 failures/errors/skips. Ten small-tensor checks cover window order, exact zero-residual readout, full ordered-head capacity, strict migration and independent trainable residual weights. Additional call-path checks cover collecting multiple comparison failures without dropping NMSE/rate, separately rejecting same-H_hat RZF replay differences even below the cross-model tolerance, retaining metric exceptions/nonfinite evidence in valid JSON, rejecting v1/v2 namespace or format confusion, preserving v1 marker bytes, and rejecting incomplete preflight acceptance. A further small-tensor schema regression verifies source optimizer evidence using the original parameter order: only the two explicitly named new residual weights are excluded for that source check, saved moment IDs/shapes/dtypes are validated, and no old optimizer state is loaded into the fresh optimizer or advanced by an optimizer step. The remaining targeted checks retain source identity, signed paired statistics and interruption/recovery coverage. All six Python files passed AST syntax checks; all three entry points passed `python -B -S ... --dry-run` without model/checkpoint loads or creation of the v2 result root. An independent AST extraction of the six comparison-tolerance dictionaries from commit `795a0cc` matched the current dictionaries exactly. The five protected pre-existing manual plotting/figure files retained their initial hashes.

No complete codec/CE model, CDL generation, training, inference, CUDA acceptance or SSH execution is performed locally. Full-checkpoint v2 functional migration, same-H_hat RZF replay, real finite nonzero residual gradients, actual complete parameter counts, and optimizer continuation remain pending server execution. Expected improvement is not recorded as an achieved result. Explicit resume can finalize a fully validated terminal 3024 checkpoint with zero additional updates when its saved receipt is intact; source and checkpoint bytes remain unchanged.
