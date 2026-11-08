# K8 RZF-matched NMSE/joint-loss continuation

## Current evidence and question

Prepared on 2026-11-08. The user reports that the frozen RZF diagnostic completed: both FDMA checkpoints selected alpha 0.03, and both G4 checkpoints selected alpha 0.1. After that adjustment, the old joint-loss models did not retain their earlier all-six-SNR mean advantage. The old small monitor still improved, which does not establish convergence across the full condition grid. These are user-provided observations; this preparation does not claim the server result package or checkpoint contents were read locally.

This batch tests NMSE versus joint task loss again while matching the training, rate-gradient calibration and monitor RZF setting within each layout. It changes neither architecture nor the RZF solver/power-normalization implementation. It does not tune alpha during training. There is no new performance result for this batch yet.

Outputs are independent under `runs/twc_v1_k8_rzf_matched_training_v1/`. Source training implementation `596d2a4d640c9001b85f78dec939bf032ff4a024`, the completed RZF diagnostic implementation `0e0b6e8d258731cf7de98f77e7fd87a470d6d6be`, ordered-window implementation `b6dc7ed5453ad0f8bc581b157505e4656932c1b7` and ancestor `8096ed965027a92f6e0738c7653b4c3e75f71b8c` retain separate identities. The new experiment commit is recorded separately rather than substituted for a source version.

## Shared NMSE sources: no old joint checkpoint initialization

Both new FDMA arms start from the same old `fdma_nmse` final state. Both new G4 arms start from the same old `g4_nmse` final state. **Neither joint-loss arm starts from an old joint-loss checkpoint.** This makes the initial state matched within each layout instead of carrying forward different earlier objectives.

| New arms | Read-only source, relative to repository root | SHA256 |
| --- | --- | --- |
| `fdma_nmse`, `fdma_joint` | `runs/twc_v1_k8_task_loss_screen_v1/training/fdma_nmse/checkpoint_latest.pt` | `5feae0f5c3d6da9386a05d6c4ff1a5d939c54a7a1ebf2ea7065e23fa46dd13be` |
| `g4_nmse`, `g4_joint` | `runs/twc_v1_k8_task_loss_screen_v1/training/g4_nmse/checkpoint_latest.pt` | `6fa136d9ce5afbc545a85d1646018dbd609df2a9b2a4e2524c42aa7c4fb67d57` |

The server verifies source step 6048, cumulative step 22096, completion, SHA256, full saved configuration/fingerprint and the provenance chain. All codec state, including learned ordered-window residual weights, is strictly loaded. There is no mean-to-ordered migration, head reset, architecture resizing or CE retraining. Source optimizer/scheduler/beta records remain historical evidence and are not loaded as the new training state.

## Fixed training recipe and addresses

| Setting | All four arms |
| --- | --- |
| System | K8 / Nt32 / Nsc256; original 256 feedback-RE protocol |
| Layout | Existing FDMA or G4; original physical mapping, UE energy/noise and prefix rules |
| Train seed / physical batch | 20261107 / 4 complete multiuser channels |
| New updates | 12096 per arm; 48384 total |
| Cumulative training addresses | `codec_train` steps 22097–34192 |
| Optimizer | Fresh AdamW, LR 2e-4, weight decay 1e-5 |
| Scheduler / gradient clipping | Constant; norm clip 1 |
| CE and network | CE frozen/eval; original encoder, Joint decoder and all codec modules retained |
| RZF | Original solver/normalization; fixed alpha 0.03 for FDMA, 0.1 for G4 |
| Technical segment | First 108 updates included in 12096, then automatic continuation |
| Checkpoint selection | Prescribed final local step 12096 only, not monitor-best |

The two objective arms within each family share initialization, data addresses, gamma/Ep schedules and standard pilot/data-noise addresses. G4 remains G4; the existing two key/shared-energy ratios and continuous cumulative phase are retained. No earlier training address is replayed as a new update. No source optimizer moment, scheduler epoch or earlier beta is continued. The read-only `runs/twc_v1_k8_rzf_probe_v1/alpha_selection.json` provides the independently selected training-alpha provenance; its identity and expected NMSE-source choices must be verified rather than rescored or rewritten. The chosen layout alpha stays fixed for training, beta2 calibration and every progress monitor; no test, monitor or later alpha sweep can change it retroactively.

## Fresh beta2 calibration

A new immutable calibration record is required. It uses the two frozen NMSE source checkpoints above, in eval mode with gradients enabled, at fresh `codec_train` addresses 500100–500111. Each source uses twelve batches of four; all six gamma conditions appear twice, Ep is 16, and G4 uses key:shared ratio 1:3. The RZF path in each source's rate gradient uses its fixed family alpha, 0.03 or 0.1.

For the pooled 24 records, let `g_N` be the full-codec gradient norm of NMSE and `g_R` the full-codec gradient norm of the unscaled mean downlink sum rate. Missing gradients contribute zero on the same parameter basis. The one shared target is:

```
beta2 = 0.25 * median(g_N) / median(g_R)
```

No optimizer is created or updated during calibration. Source parameters/buffers and CE remain unchanged. This is independent of the old task-loss beta and old alpha-selection record; neither is relabeled as beta2. Zero/nonfinite norms or incomplete source/data evidence are rejected. The complete accepted record, identity and SHA256 bind the new configuration/checkpoints and strict resume. A partial attempt is preserved, not silently recalculated or overwritten.

For a new joint arm at local update `step`, use:

```
L = L_nmse - beta2 * min(step / 756, 1) * mean_downlink_sum_rate
```

The first 756 ramp updates are part of the 12096 budget. There is no Rref or extra rate normalization. The NMSE control keeps rate weight zero. The shared beta2 is not retuned separately for Hybrid or FDMA or after looking at new performance.

## Four full-grid monitors, not checkpoint selection

Monitor at local steps 3024, 6048, 9072 and 12096 using the same fixed 64 complete multiuser validation channels: 16 batches of four at addresses 8960–8975. Evaluate all six gamma points; G4 includes both ratios 1:3 and 1:1 with equal weight. FDMA is evaluated once per gamma. Report objective components and the complete condition grid with the family-fixed RZF alpha.

Monitoring temporarily switches modes without contaminating training: restore the original model/CE modes and all relevant external RNG state afterward. The data protocol remains address-based. Monitor data are distinct from beta2 calibration, final alpha calibration and final validation. Store 16 small monitor reports, one for each of four models at four prescribed steps; strict resume retains and verifies earlier reports. These full-grid observations improve on the old gamma10/Ep16 small monitor but do not prove convergence. They do not select best, shorten or extend the 12096 updates, retune alpha or create an additional series of large intermediate checkpoint copies.

## Final fixed-alpha and newly tuned-alpha comparisons

Evaluate six frozen checkpoints on common actual channels and standard pilot/data noise: four new final models plus the two old NMSE sources. The source predictions are shared within a batch; the sources are not copied into the new output tree or evaluated repeatedly to manufacture more samples.

Keep two separately declared evaluation policies:

1. **Training-matched fixed alpha:** 0.03 for every FDMA checkpoint, 0.1 for every G4 checkpoint.
2. **New final-only selected alpha:** one alpha per checkpoint, selected from the predeclared grid below using independent calibration channels. This is an evaluation diagnostic and is never written back into a training recipe, training checkpoint or monitor history.

```
[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
```

Final-only alpha selection uses 64 complete multiuser validation samples, batches 6144–6159, with batch size four. Each of the six checkpoints receives a single global alpha chosen by mean downlink sum rate over all six gammas; G4's two ratios are included with equal weight. Exact-score ties choose the smaller alpha. Save all scores and boundary flags in a new non-overwriting record; do not reuse the old probe's selected record, select per SNR/ratio/sample or change the grid after seeing results.

After selection is locked, final validation uses 256 complete multiuser channels, batches 7168–7231. It cannot influence the alpha choice. All six gamma points, Ep16, DL25 and both G4 ratios remain visible under both fixed/tuned policies. Neural H_hat is computed once per batch/model/physical condition and reused for the candidate alphas or fixed/tuned policies. No persistent CSI/H_hat cache is written. Raw reconstruction NMSE/norms cannot change with alpha when H_hat is held fixed; true H_dl is only an offline scoring target and is not used for per-sample amplitude fitting.

Preserve sample-level and batch-level paired means/SE/95% intervals and all negative outcomes. Across-gamma summaries average within the complete multiuser sample before estimating uncertainty. Batch size is four; users, subcarriers, ratios or gamma rows do not multiply the independent sample count. Compare joint versus NMSE within each layout, every new model versus its corresponding source, and the Hybrid–FDMA gap and its objective interaction. The single-training-seed intervals describe channel-sampling uncertainty conditional on these checkpoints and the locked selection, not training-seed variability or uncertainty in alpha selection. Do not choose the more favorable fixed/tuned policy retrospectively.

Test remains sealed. This is another controlled validation screen, not final journal evidence or proof that either layout or objective must win.

## One server batch and strict recovery

The controller runs source verification, new beta2 calibration, four 108-update technical segments with automatic continuation to 12096, final-only alpha selection/validation and compact results. Necessary identity, nonfinite, power, freeze or data-pairing failures preserve evidence and stop with nonzero status. Completing the first 108 updates never adds budget or resets a scheduler.

Explicit resume requires the exact arm, complete configuration/fingerprint, source identity, beta2 record/hash, model, fresh-run optimizer, scheduler, RNG and cumulative data address. It cannot fall back to initialization, substitute an old joint source, regenerate an accepted beta2 or change training alpha. A rolling digest binds each training step's actual channel fingerprint and prescribed noise address. This does not claim that every training noise tensor is fully hashed; saved seeds/addresses and actual monitor noise fingerprints retain their separate meanings. Completed stages are reused only after evidence/hash verification; failed attempts and original source evidence remain intact.

Each arm keeps one rolling large `checkpoint_latest.pt`. At completion, that same file is the prescribed 12096-step final state, with a small terminal JSON/SHA alias rather than an independent large final-checkpoint copy. No best checkpoint is selected.

From the 03 repository root in the existing CUDA/Sionna environment:

```bash
python -B -u run_twc_v1_k8_rzf_matched_training.py --run
```

Read the complete status/result handoff:

```bash
python -B run_twc_v1_k8_rzf_matched_training.py --summary
```

Explicit continuation after interruption:

```bash
python -B -u run_twc_v1_k8_rzf_matched_training.py --run --resume
```

The controller performs beta2 calibration and final evaluation internally. Summary reports stage/error/unknown state, per-arm current step and objective components, beta2 identity, four full-grid monitors, final fixed/tuned pointwise results and paired intervals, selected-alpha edge flags and exact small-package paths. It does not load large weights just to display progress. Source weights and full raw samples stay at their recorded server paths. No paper or historical experiment is rewritten.

## Actual local checks and remaining server validation

Executed locally on 2026-11-08: `python -B -m unittest tests.test_twc_v1_k8_rzf_matched_training -v` ran 20 tests with 20 passes and 0 failures/errors/skips. An independent `-B -S` run passed 15 standard-library checks and explicitly skipped five checks requiring the real Torch/NumPy dependencies. All five new Python files passed AST syntax checks, and all three entry points passed standard-library `-B -S --dry-run` checks. The new output root was not created. A Git diff against `0e0b6e8d258731cf7de98f77e7fd87a470d6d6be` confirmed all 33 pre-existing files in the old probe's `SOURCE_FILES` list remained unchanged.

The numerical monitor test uses a real small neural module and actual Python, NumPy and Torch CPU default RNG streams. It verifies restoration after both normal and exceptional exits. Expensive channel/forward work is replaced by an explicit synthetic fixture, and the CUDA RNG-state list is empty in that test. It is not acceptance of CUDA RNG restoration or the real 64-channel GPU monitor. Strict source-copy checks use a small module and retain a nonzero residual parameter; they do not load a server model. Further lightweight checks cover the old-alpha identity, the two-NMSE-source beta2 protocol, condition counts and the controller's actual 108-to-12096 inspection/verification path with small state schemas.

Local preparation does not load real checkpoints, instantiate the full codec/CE, generate CDL, run model evaluation/training or connect SSH. Missing dependencies are not installed or fabricated.

Server work remains for real source/provenance verification, strict state loading, the new gradient calibration, CUDA numerical/memory behavior, each complete training/monitor/resume path and final selection/validation statistics. The prior diagnostic findings remain user-provided observations rather than locally rerun results. No beta2, new improvement, final selected alpha or convergence conclusion is claimed before that evidence exists.
