# K8 ICC-aligned handoff

This independent batch is prepared locally. It has not run on the server and
has no performance result, trained checkpoint or accepted CUDA admission yet.
The original STATUS and all historical training/finish sources remain unchanged.

- Task: fresh ICC-mechanism K8 adaptation, matched FDMA24 versus G4 key4/shared40.
- Confirmed source: read-only ICC `b8ea3a51b051b7aae22916d99aa52d811edb0850`;
  file hashes and module mapping are in `TWC_V1_K8_ICC_ALIGNED_REFERENCE.md`.
- New mechanisms: 32 ordered Swin-window features, common216-complex interface,
  active-vector joint Ed48 normalization, rank8 Fourier-LS plus new pilot-only
  residual U-Net, causal per-coordinate EMA prior, calibrated per-UE/per-RE CE
  error, and full two-Joint/spatial-cross/four-Swin physical CSI reconstruction.
- Fixed budget: micro4 x accumulation8; CE5000 updates and160000MU frames;
  each codec20000 updates and640000MU frames/5120000UE inputs. The two arms
  share640000 actual replayed MU frames, not1280000 independent frames.
- Metric protocol: monitor128MU, final1024MU, gamma0/5/10/15/20/25, alpha.03,
  DL25, single seed. Optional alpha tuning is not enabled; its address range is
  reserved and unconsumed. Old test remains sealed.
- Local verification is limited to standard-library checks and tiny CPU
  components. Complete model/CUDA admission, real accumulation throughput,
  peak memory with AdamW state, CE training/calibration, codec training/resume
  and final paired performance still require the server.
- Actual final local suite:43 tests passed,0 failures/errors/skips. A separate
  `-B -S` run passed22 and explicitly skipped21 real Torch/NumPy-dependent
  tests. Four CLI dry-runs and the no-start summary passed without creating
  the new output root. Numerical tests used only tiny CPU tensors/modules.
- Next single action: safely synchronize03, activate the existing precoder
  environment and launch `run_twc_v1_k8_icc_aligned.py --run` once. Return its
  complete `--summary` after completion/failure. Resume is explicit with
  `--run --resume`; completed hash-verified stages are reused.

Full configuration and commands: `TWC_V1_K8_ICC_ALIGNED.md`. No local training,
CDL generation, full-model execution, checkpoint download or SSH connection
was performed. No benefit over FDMA is assumed.
