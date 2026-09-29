# K4 low-feedback-SNR bottleneck diagnostic

This is **configuration-selection validation**, not another formal held-out test.
It launches no training and does not change models, RZF, noise conventions,
checkpoints, formal CSVs, or paper figures. Do not repeat the completed 20k
FB~U(0,25), DL=25 pure-rate refinement. Any further <=5k refinement is a separate
decision after reviewing these diagnostics.

## Fixed protocol

- K=4, Nt=32, Nsc=Dtot=256, four subbands, current evaluator's CDL-C channel.
- Five existing 50k allocation specialists: (64,0), (48,64), (32,128),
  (16,192), (0,256); existing 20k-refined robust pure-AS; task-trained Swin and
  CsiNet+ at (64,0). Eight fixed configurations, no universal model or oracle.
- FB=[0,5,10,15,20,25] dB; DL=25 dB; unchanged physical UL slicing.
- Validation seed **20360928**, batch size 8, **256 channels** (32 batches).
  This is separate from formal test seed 20260921 and training validation seed
  500042. Do not merge these results into the formal paper artifacts.
- First use generates and saves the actual H_dl/H_ul arrays (about 128 MiB)
  under ignored runs_as_screen/diagnostics/k4_low_feedback_snr/.
  Subsequent runs reuse that exact cache, verified by SHA-256, not merely a seed.
  An incomplete or inconsistent cache raises an error; it is never auto-deleted.
- --preflight evaluates the first eight of the SAME 256 cached channels; on
  first use it still creates the complete cache. Every configuration/SNR uses
  the same batch tensors. Feedback seeds follow the existing evaluator; different
  stream shapes do not imply identical noise tensors.
- All weights use the existing strict=True loaders. Supported allocation and
  running-power index stay separate from full architecture widths (64,256).
  Actual loaded training configurations, selected checkpoint steps/paths/hashes,
  software versions and source hashes are recorded. Incomplete/unknown run
  status or an incorrect training recipe stops the diagnostic.

## Server commands

Run these yourself after SSH login; this tool never connects to the server.
The expected checkout is ~/project/03_PRECODER, in the precoder environment.

First inspect and update without discarding local work:

~~~bash
conda activate precoder
(
  set -eu
  cd ~/project/03_PRECODER
  test "$(git branch --show-current)" = main
  test "$(git remote get-url origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git
  test "$(git remote get-url --push origin)" = https://github.com/Charlie-Su-55/03_PRECODER.git
  git status --short --branch
  git diff --cached --name-status
  test -z "$(git diff --cached --name-only)"
  git fetch origin
  git log --oneline origin/main..HEAD
  test "$(git rev-list --count origin/main..HEAD)" -eq 0
  git pull --ff-only origin main
  git status --short --branch
)
~~~

Stop on any error, divergence, staged content, or conflict. Never reset, clean,
stash automatically, force-push or overwrite server changes. The pull itself
refuses conflicting tracked/untracked files; non-conflicting local work can stay.

Then run in order, stopping on any error:

~~~bash
cd ~/project/03_PRECODER
python -m unittest discover -s tests -p test_low_feedback_diagnostic.py -v
python diagnose_low_feedback_snr.py --dry-run
python -u diagnose_low_feedback_snr.py --preflight
python -u diagnose_low_feedback_snr.py
~~~

--dry-run reads only checkpoint existence/completion metadata; it does not load
weights. The last two commands require the GPU server. Preflight now runs all
48 method/SNR conditions on eight channels in three sequential trials: Plain A,
Plain B, Probed C. Each gets a fresh CachedBatchSource/cursor/batch_state and the
same initialized model/RNG state. It checks the plain replay control BEFORE
interpreting a probed difference. Both tolerances remain 1e-6.

The existing channels.npz/channels.json are read and hash-checked, never deleted,
recreated or overwritten by this fix. Every attempt gets a new timestamped
output directory, including failed attempts. If preflight fails, stop; send back
probe_transparency_report.json before running the 256-sample diagnostic.
No training command is part of this workflow. No new checkpoint is produced.
--print-plan is a CPU-only preview without checkpoint checks or torch imports.
Each evaluation uses a new timestamped directory; existing outputs are refused.

## Outputs and interpretation

Inside the printed result directory:

- summary_low_feedback_validation.csv: **48 rows**, per-point rate mean/SE/CI
  and existing raw reconstruction NMSE. Preflight uses a _preflight tag.
- samples_low_feedback_validation.csv.gz: all per-channel rates and diagnostics;
  retain negative outcomes, not just favorable allocations.
- low_snr_validation.csv: eight rows; per-channel average rate over 0/5/10 dB,
  then mean and SE across 256 channel indices. Signed paired mean differences
  and SEs versus Swin, CsiNet+ and original pure-AS are retained. The 3*256
  correlated SNR observations are NOT treated as independent samples.
- physical_link_batches.csv: actual received clean power, expected complex
  noise variance (and variance per real component), realized noise power, and
  measured pre-receiver SNR for every active branch/batch/condition.
- channel_sample_index.csv: sample/batch/within-batch indices, generation seed
  and channel-cache hash.
- diagnostic_manifest.json: checkpoint paths, steps, hashes, actual saved
  training modes/configs, seed, dataset role, source hashes and metric definitions.
- probe_transparency_report.json: preflight A/B/C comparisons, RNG/memory/cleanup
  evidence and backend settings, saved even when metric comparison fails.
- diagnostic_complete.json: success marker; absent means the run is incomplete.

All result CSVs are marked as validation/preflight as appropriate. This script
never writes paper/data/, paper/figures/ or overleaf/. No row-wise maximization,
per-channel allocation selection, winning-only filtering, automatic training
selection or oracle deployment curve is generated.

## What the physical probes mean

The original code already exposes physical tensors at add_noise /
add_relative_awgn and the baseline fdma_feedback_mrc boundary. The standalone
probe temporarily wraps these callables and registers forward pre-hooks; each
original operation is called once and its original output is returned unchanged.
No random values are drawn by the probes. Wrappers/hooks are removed on exit.
Model/source files, state dictionaries, normalization, receiver and slicing
operators are unmodified. No LMMSE demixer is introduced.

**Horizontal-axis SNR:** let P be the mean squared magnitude of the *clean*
received stream, averaged over the whole current batch and all active entries.
The existing code uses expected complex noise variance P*10^(-SNR/10), with half
that variance per real component. This is a stream-relative BS-antenna input
SNR, before MRC, antenna DFT or learned decoding, not a common absolute noise
floor, transmit Eb/N0 or a neural post-detection SNR.

- FDMA averages B*K*Nt*Df received elements; physical user-k slice remains
  [k*Df:(k+1)*Df].
- AS averages B*Nt*Da elements AFTER the coherent sum of all four UEs on
  [K*Df:K*Df+Da]. The probe checks the actual noise input against this full sum.
  It records both its power and the sum of individual user powers, so the
  coherent cross-term contribution is explicit.
- Expected variance is the generator parameter, whereas realized noise power
  is measured from the actual noisy-minus-clean tensor. They need not be equal
  for each finite batch. Inactive branches have no physical rows, not zero-noise
  observations.
- Baseline MRC diagnostics record the actual normalized transmit-symbol power,
  output symbol MSE/NMSE, clean-channel error, and projected actual noise power.
  Expected post-MRC noise uses the existing combiner and denominator (with the
  same clamp). These refer to physical symbols, not the decoded channel or an
  arbitrary neural feature. The proposed decoder has no analogous explicit
  symbol detector; no invented post-detection SNR is reported for it.

These data can test whether a different fixed allocation is a better low-SNR
validation candidate, whether input-noise power scales with coherent AS power,
and how much noise MRC removes for FDMA baselines. They do not by themselves
prove a causal architectural bottleneck or a formal test/SOTA advantage.


## Preflight transparency report (2026-09-29 repair)

The previous checker compared only Probed versus Plain, zipped row order, and
raised at the first metric mismatch. Consequently its subsequent RNG comparison
was never reached. A disk checkpoint hash could not detect in-memory buffer
changes. The supplied RTX 5090 log establishes that both evaluations completed,
but contains no failed sample key, difference magnitude or RNG/state evidence.
It therefore does NOT establish a GPU root cause.

Read-only call-chain findings:

- baseline_evaluate.set_seed seeds Python random, NumPy global state, Torch CPU
  and all CUDA devices. evaluate_scenario resets condition seeds and reuses
  the channel batch; cached replay bypasses CDL/Sionna generation entirely.
- The proposal writes running_pwr only inside its training branch. Eval reads
  frozen running power. DFT matrices are registered buffers; no explicit
  persistent forward cache was found in the inspected application code.
- Baseline BatchNorm modules use eval mode. CsiNet+ has in-place ReLU on internal
  activations, not a normalization-state update or demonstrated cache mutation.
  normalize_complex_symbols is stateless, returns a new divided tensor, and
  does not modify its symbol input in place.
- The noise functions draw via torch.randn_like without an explicit generator.
  No independent generator was found in the cached evaluator/model call path.
  The unused pilot-estimation path accepts an explicit generator, but is not
  called here. Runtime discovery also checks model attributes and evaluator
  globals for Torch generators, NumPy Generator/RandomState and random.Random.
- The evaluator's h_estimator forward hook is removed in a finally block. Probe
  wrappers already used ExitStack, but cleanup was not previously verified.
  Normal and exceptional exits are now checked for exact wrapped-callable and
  hook restoration, including absence of newly shadowed instance attributes.
- utils.__init__ calls setup_gpu at import and the diagnostic calls it again;
  this explains two setup banners, not the numerical discrepancy. Snapshots
  are taken after all imports, model loading and channel-cache initialization.

What the repaired preflight records:

- Python/NumPy/Torch CPU/all CUDA RNG snapshots and supported independent
  generator states. Initial states are restored for each trial. End-state
  fingerprints are compared by RNG family/device; equality is evidence, NOT
  proof that identical noise was drawn at every location.
- Model parameters and all registered buffers, including non-persistent ones,
  module identities/training flags, unregistered tensor/cache attributes, the
  in-memory channel cache and the actual supplied batch tensors.
  Opaque application-cache objects are listed as unverified and fail the state
  coverage check rather than being silently accepted as unchanged.
- Changes are captured BEFORE any restore. A changed buffer/mode/input causes
  failure even if restoring it makes the next replay numerically match.
  Snapshots are CPU copies; there remains only one set of GPU model objects.
- Complete sample keys: scenario_id, method_key, allocation_df, allocation_da,
  feedback_snr_db, downlink_snr_db, sample_index. Reordering is allowed; missing,
  unexpected, duplicate and non-finite entries are failures.
- For rate/NMSE/precoder power: maximum absolute/relative differences, counts
  exceeding tolerance, first and maximum-difference keys, and A/B/C values.
- Hook/callable restoration, evidence-collection errors, runtime exceptions,
  and numerical backend settings (TF32, matmul precision, cuDNN, SDPA,
  deterministic flags and relevant environment variables).

The JSON is written before raising on failed comparisons. Plain A/B failure is
labelled "baseline replay not reproducible", not "probe changed sum_rate".
If A/B passes but A/C fails, affected method/SNR/sample keys and wrapper/operator
paths are provided as localization candidates, not as an asserted cause.

No new deterministic/precision mode is enabled. The existing "high" matmul
setting is retained; native CUDA/library algorithm caches are not reset.
Unknown backend state cannot be ruled out by application-state snapshots alone.
The tiny CPU regression tests cover replay controls, RNG/buffer side effects,
report-on-failure, sample pairing and exceptional probe cleanup. They do not
replace the real-checkpoint GPU preflight, which still needs to run on the server.
