# ICC mechanism to K8 implementation record

This is a new K8 adaptation, not an unchanged ICC experiment or a transfer of
08 weights. Source was actually read using `git show` at the fixed training
commit `b8ea3a51b051b7aae22916d99aa52d811edb0850`. The 08 working branch was
not changed. The later documentation reference is
`bfa04279dd55873a61ecdd476c0144a8025cd496` on `research/step4h-r1`; 08 main,
Step4G, its historical128-RE/SIP protocol, and its weights are not the source
of this experiment.

## Mechanism mapping

| Pinned ICC module | New 03 implementation | What is retained / adapted |
|---|---|---|
| `round6_resource/codec.py` `OrderedWindowEncoder` | `models/twc_v1_k8_icc_aligned.py` `K8ICCAlignedEncoder` | Full32x256 angular-delay CSI; patch4; four width128, heads4, window4 non-shifted Swin blocks; LayerNorm; mean inside32 actual4x4-token windows in delay-major/angle-major order; ordered4096-vector to432-real common216-complex interface. No global-mean bottleneck or delay truncation. |
| `shared/swin_codec.py` / `baselines_03/transforms.py` | Existing 03 `models/swin_precoder.py` / `models/baseline_common.py`, locked as dependencies | The corresponding patch/window attention, GELU MLP, real/imag layout and full angular-delay transform/inverse operations. These are codec numerical modules, not an import of 08's physical system. |
| `round8/physics.py`, `ul_unet/receiver.py`, `ul_unet/statistics.py` | `utils/twc_v1_k8_icc_aligned_physics.py` | One shared noisy MAC, paid pilots, Fourier rank8 LS with actual amplitudes, jointly normalized selected data vector, corrected-channel statistical recovery, per-coordinate historical power and per-RE independently calibrated CE error. Exact physical audit/receipts live with the new batch; these modeled error/precision features are not calibrated posterior confidence. |
| `ul_unet/estimator.py` `PilotResidualUNet` | `K8ICCAlignedPilotUNet` | UE-shared133-channel frequency residual U-Net, widths32/64/96, GroupNorm8/GELU, two down/up stages, head Normal(0,1e-4) with zero bias. Only pilot-supported Y, known pilots, Fourier initial CE, masks and public SNR are inputs. Output is full-frequency correction without rank8 reprojection. |
| `round4_joint_decoder/decoder.py`, `round6_resource/decoder.py`, `round8/decoder.py` `JointDecoder` | `K8ICCAlignedJointDecoder` | Two width128/heads4 pre-norm all-user self-attention → observation cross-attention → FFN256 GELU layers; physical resource overlap and estimated-channel coupling biases; small nonzero latent residual head;432→65536 FC per UE; spatial position/user identity; global spatial cross-attention; four Swin blocks and complete256-tap reconstruction/inverse. |
| `ul_unet/model.py` `make_models` | `create_codec` plus server common-initialization admission | A shared fixed initialization seed849301; exact common parameter values, independent copies/storage and optimizers. Both arms start fresh rather than inheriting diverged historical FDMA/G4 weights. |

## Explicit differences from the pinned ICC run

- K4 becomes K8, retaining Nt32/Nsc256 and the existing03 CDL-C4x8 array.
- Total256 RE contain64 exclusive pilot RE (8 per UE). FDMA data are24
  coordinates per UE. G4 has four fixed2-UE groups,4 key coordinates per UE
  and40 shared RE per group. The common neural interface remains216.
- Per-UE energy64 has Ep16/Ed48. Only the selected24 or44 coordinates are
  jointly normalized to48. There is no forced1:1 or1:3 key/shared split.
- Physical `N0=.25*10**(-gamma/10)` remains03's convention; ICC's `.125`
  coefficient is not copied. Each frame has one complex noise realization
  with `E|n|²=N0`, shared by its complete MAC observation.
- Training gamma expands from ICC's5–25 to0–25 dB. The corresponding neural
  feature is explicitly `(gamma-12.5)/12.5` instead of ICC's
  `(gamma-15)/10`. This feature does not modify physical noise.
- Pilot features include actual known sqrt(16/8) amplitudes. The separate
  Fourier design checks its actual paid8-pilot rank, unit-column basis,
  Gram diagonal16/256 and ridge `.001` (ridge/diagonal `.016`).
- K8 memory projections consume512 real estimated-UL features and16 real
  pilot features per RE; query feature width19 and spatial-position width14
  include8-user identity. There are1728 padded codeword queries and4096
  spatial queries. Chunked SDPA preserves every key/value and global user
  communication; it does not replace attention with per-user/local attention.
- Current-frame key/shared energies are offline diagnostics only. Receiver
  functions accept no true UL/DL, transmitted codeword or current-frame power
  side information. The new decoder receives one measured Y, the same
  corrected UL used in recovery, analytical soft values/reliability, public
  pilots/SNR/layout. Its unconstrained residual output is a reconstruction
  latent, not a physically renormalized transmission or a new residual loop.
- Source ICC K4 batch64×20k exposes1,280,000 MU frames/5,120,000 UE inputs.
  New K8 effective32×20k exposes640,000 MU frames/5,120,000 UE inputs per
  codec. Equal UE exposure is not equal MU-frame or compute budget.

## Raw archived source SHA256

All paths below are under `src/icc08/` at the pinned training commit. Digests
are computed from raw `git show` blob bytes, not Windows checkout newlines.

| File | SHA256 |
|---|---|
| `resource_research/round6_resource/codec.py` | `843b400676699d194f456461fe468a535c47ea8f929ac33241cb6e41606ab8d1` |
| `resource_research/round6_resource/decoder.py` | `fed46bad6071dc7bc6cc9c11031a2c4406f292f4eb5785038756b53e9aa28590` |
| `resource_research/round8/decoder.py` | `b4ac804d7aa0d6d0f2da2566801cd6a70f0b5daa84c8bfd68518547160968124` |
| `resource_research/round4_joint_decoder/decoder.py` | `1997cc57c1003c4239861f84c30dfb92ff38f4ef24f2566de404baac8ea62d25` |
| `resource_research/ul_unet/estimator.py` | `9bdf18e8af0db97c5c46e560f83760170a3762cf85aa39832e8681f1fea577ce` |
| `resource_research/ul_unet/model.py` | `8f97a4886faaf474271da62d799fc2e0dc0bc4c542916f5aef3b1af34148aa09` |
| `shared/swin_codec.py` | `08249826ea8a01832a6ebcfda8aa1e0d6358b2863500f64be3258aa1f3875d78` |
| `baselines_03/transforms.py` | `d71429e9b623784a354fa87bf9541763d6ed325ad998258998acaf94c7c10f0b` |

## Model-component validation boundary

Local execution built no full codec and generated no CDL. Seven small CPU
model-component tests passed: ordered-window geometry;216-slot resource map;
new CE features ignore data-only NaN/Inf and have zero data-RE gradients;
tiny all-user Joint attention has finite memory/other-query gradients;
independent copies/strict tiny loading; public decoder callable excludes
truth/current-frame energy inputs. Five light admission-helper tests passed.
In standard-library-only mode the3 admission-plan/CLI tests passed and the9
Torch-dependent component checks explicitly skipped.

`preflight_twc_v1_k8_icc_aligned.py` is prepared but was not run locally. On
CUDA it builds the full common fresh models on CPU, verifies parameter
equality/storage independence, and moves only one codec at a time to GPU.
It checks syntheticB4 physical energy/noise, actual Fourier design,
pilot-only CE, strict fresh-state loading, required module execution, DL-NMSE
backward gradients, frozen CE/prior and original normalized RZF power. It
performs zero optimizer updates and uses explicitly synthetic CE-error
features; it is not independent CE calibration or a performance result.
Training memory/throughput/ETA must come from the included first100 real
updates with full AdamW state. No parameter count or runtime gain is asserted
without the server instantiation/report.
