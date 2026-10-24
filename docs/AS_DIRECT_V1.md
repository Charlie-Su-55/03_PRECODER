# AS user-message direct path (as_direct_v1)

## Scope and sole architectural change

`models/adaptive_hybrid_as_direct.py` defines the independent, opt-in
`AdaptiveHybridPrecoderASDirect`. Its base is **Full**
`AdaptiveHybridPrecoder`, not Lite1024. No existing model, configuration,
training/evaluation entry point, checkpoint, result, or paper file is changed.
Reference code: `a2c3707f4a8c388d3cfb68058a0523d2b45c3d5e`.

Each original refinement layer already computes a per-user message after
`user_query_attn` and `msg_proj`, before summing over users. The new adapter
returns this same tensor together with the original shared coupling. It keeps
all original bias/attention/FDMA-injection/cross-user/FFN/norm operations, then
adds, **after the original layer's final output**:

```text
F_new = F_old + tanh(alpha_l) * msg_k[:, :, None, :, :], if d_a > 0
F_new = F_old,                                        if d_a == 0
```

The returned message is `[B,N_sb,K,D]`; permuting it to `[B,K,N_sb,D]`
and inserting the antenna axis broadcasts it over `Nt`, with no sum or
averaging over users in the new path. The original user sum is retained.

- One zero-initialized scalar `layers.<index>.alpha` per refinement layer.
  Static accounting: `L=5` adds **5 parameters**, with **0 UE parameters** added.
  No existing parameter changes shape or value during wrapping.
- The top-level forward explicitly passes `as_active=(d_a > 0)`; a pure-FDMA
  synthetic global context does not enable the new path.
- AS-active forwards always evaluate the gated addition, including at zero
  gate. Thus the gate derivative is not removed by a zero-value shortcut.
  The direct contribution to message gradients is naturally zero at zero gate;
  its connectivity is separately tested with a nonzero gate.
- No new projection, normalization, attention, detach, hook, message cache,
  analytical demixer, noise convention, or loss is introduced.

Construction first runs the original constructor, then re-registers its
already-initialized layer children under the same names. It does not draw new
random weights. Unexpected direct layer parameters/buffers or child-module
sets are rejected instead of silently dropping future original-model state.
Only the local GNN/refinement/top forwards are reproduced where explicit
message/activity arguments are necessary. AST regression checks constrain
these copies to the allowed differences. Encoder, transmission, padding,
noise, and RZF implementation are inherited. `H_ul_est`, running power,
prefix selection, legacy training return `(W_train,H_hat)`, and evaluation
return `W_eval` keep the original contract. Neither default factory nor
optimizer behavior has been changed.

## Explicit old-weight migration versus normal restoration

```python
from models.adaptive_hybrid_as_direct import (
    AdaptiveHybridPrecoderASDirect,
    migrate_original_state_dict,
)

# cfg must preserve the source architecture maxima AND its supported grid.
model = AdaptiveHybridPrecoderASDirect(cfg)
# old_state is an already obtained Original/Full tensor state_dict.
# This snippet performs no checkpoint file access.
migrate_original_state_dict(model, old_state)
assert model.architecture == "as_direct_v1"

# A subsequent ASDirect state must be restored normally, without migration:
model.load_state_dict(as_direct_state, strict=True)
```

`migrate_original_state_dict` checks the complete original key set, every
tensor's shape, and every dtype **before loading any tensor**. It rejects
missing/unexpected keys, malformed tensors, dimension mismatch, and dtype
mismatch. It adds only the explicit gate-key whitelist with zero values,
preserves state-dict version metadata, and calls `load_state_dict(strict=True)`.
It does not modify the supplied state, crop weights, remove prefixes, drop
parameters, or silently convert dtypes. Gate-bearing ASDirect state is not
accepted as Original state. This function is weights-only initialization, not
optimizer/scheduler/RNG resume.

For a K4/Nsc256/Dtot256 pure-AS specialist, keep maximum dimensions `(64,256)`
while the supported allocation grid is only `[(0,256)]`. Its `running_pwr`
remains shape `(1,)`, with `cfg_idx=0`. No allocation-specific width inference
or grid-to-buffer conversion is performed by the adapter. A wrong config is
rejected through strict key/shape checks, not repaired automatically.

Missing gates in a normal ASDirect restore are an error. The unchanged
`main_train.py` `init_checkpoint` branch strictly loads its supplied old state;
passing a new `model_factory` alone does **not** make it a migration entry.
A future separately authorized entry must explicitly perform migration before
optimizer creation and save `architecture_metadata` plus the complete source
configuration. This task does not add or enable such an entry and does not
change old fingerprints or checkpoint formats.

## Initial local validation at 41fc81f

Local environment: Python 3.12, PyTorch `2.6.0+cpu`; normal original-model import
requires unavailable `sionna.phy`. No dependencies were installed, stubbed, or
bypassed. No real checkpoint was downloaded/loaded, no CDL channels generated,
and no training, formal evaluation, GPU profiling, or paper update was run.

Performed static checks:

- Original versus new top-forward AST: the only operational difference is
  `as_active=(d_a > 0)` on the refinement call.
- GNN AST: only the return exposes the existing per-user message in addition
  to the original coupling; the message and coupling computation are unchanged.
- Refinement AST: original operations are unchanged, apart from unpacking that
  message and the final AS-active residual.
- Original source hash matches the pre-change source. Raw Windows SHA-256:
  `efaad80d29d238326bcdd8a7e8e4212705a00d7a2edba1bff8bfb565a836ad25`.
  Portable LF SHA-256:
  `67dd73c7720a2aa7c4e33b0f5dbaa30e7824c3aa538fba6f5f73739ecd951ceb`.

CPU synthetic acceptance tests are in
`tests/test_adaptive_hybrid_as_direct.py`. The local run completed with
**18 discovered tests: 6 static checks passed, 12 numerical checks skipped,
0 failures/errors**. Both new Python files also passed in-memory syntax
compilation, without generating bytecode files. Numerical tests must be rerun in
the existing dependency-complete environment; they are **not locally verified**:

```bash
python -X utf8 -B -m unittest discover -s tests -p test_adaptive_hybrid_as_direct.py -v
```

The tests use in-memory synthetic state and small synthetic channel tensors,
not experiment channels or saved weights. They check zero-gate `H_hat`,
normalized deployment RZF and eval outputs at pure-FDMA/hybrid/pure-AS,
`H_ul_est` interfaces, nonzero-gate pure-FDMA invariance, nonzero finite gate
gradients and message connectivity, exact parameter/buffer preservation,
singleton/max-width separation, and strict migration/restoration failures.
They do not perform optimizer steps. Fixed tolerances are
`rtol=1e-6, atol=1e-6`; no precision/noise/backend change is used to obtain a
pass. Numerical replay prints actual maximum absolute differences.

**Local zero-gate maximum error: not measured (dependency skip), not zero.**
At that initial local stage, runtime parameter counting, strict synthetic loading,
gradients and numerical replay were pending. The subsequently reported server
CPU acceptance is recorded below. Real-checkpoint/CUDA behavior is still
unverified. No performance gain is claimed.

## Reported server CPU acceptance

For accepted code `41fc81f8d370d3f52b5a8910dafbb32abad96a32`, the user reported
that `tests/test_adaptive_hybrid_as_direct.py` completed on the server with
**18 passed, 0 failures/errors/skips**. The maximum absolute and relative
differences were both **0** for `H_hat`, `W_train`, `W_eval`, and
`deployed_RZF` with zero gates. Log reference, relative to the server's
03_PRECODER checkout: `terminal_logs/as_direct_cpu_OZ1waP0M`.

This is the user's server acceptance record, not a log independently fetched
in this task. It covers the synthetic CPU tests, not a real-checkpoint CUDA
run. The original local dependency skips above remain a historical record;
the complete model test suite was not rerun on the laptop this round.

## Independent real-checkpoint CUDA preflight

`preflight_proposal_as_direct.py` is a preflight tool, **not a training entry**.
No arguments show help. `--dry-run` uses the standard library only: it reads
the small archived manifest, but does not import models/Torch/Sionna, open
checkpoint bytes, test checkpoint availability, or create result directories.

```bash
python -B -S preflight_proposal_as_direct.py --dry-run
# Only in the existing CUDA environment, from the 03_PRECODER checkout:
python -u preflight_proposal_as_direct.py --run
```

The source is exactly `robust_specialist` in
`paper/data/diagnostics/k4_low_feedback_snr/validation256_cudnn_deterministic_20260930T165938_038117Z/diagnostic_manifest.json`.
Its recorded checkpoint SHA-256 is pinned to
`e21ea1feae595e0898b6be724f7062b05c09b2d1ef03fe9ffec7ac91869064bf`, with
selected step `20000` and completed robust-refinement training. The default
path is the recorded checkpoint path; it is not replaced by a search or a
different checkpoint. `--checkpoint PATH` can relocate the **same bytes**;
relative overrides are resolved from this script's repository root. Hashing
and deserialization use the same open binary handle. File hashes are checked
again after the run. No new weights are saved.

Before model state loading, the tool checks the complete saved config against
the manifest (normalizing JSON-equivalent tuples/lists only), saved step and
allocation grid. It maps the archived dimensions to a runtime namespace:
K4/Nt32/Nsc256/Dtot256/Nsb4, Full UE FFN2048, maximum widths `(64,256)`,
singleton `GAMMA_GRID=[(0,256)]`, `running_pwr.shape=(1,)`, and `cfg_idx=0`.
The full architecture grid is not substituted for the supported singleton.
Original uses `strict=True`; ASDirect uses the explicit migration helper,
which also loads strictly. The preflight checks that the only additions are
five zero scalar gates and all other source parameters/buffers are unchanged.

`--run` fails without CUDA, with no CPU fallback. It uses exactly two synthetic
complex64 multi-user samples of the formal tensor size, never CDL or held-out
experiment data. The fixed master seed is `20261024`; channel tensors are
shared across FB SNR `0/10/25` dB, with DL SNR fixed at `25` dB. Train-mode
paired replay/backward use seed `20261025`; eval-mode pairs use `20261026`.
Each pair restores Python/NumPy/Torch CPU/all-CUDA RNG states and begins with
the same loaded buffers. Train and eval are compared separately, not to each
other. Training forward's expected `running_pwr` update is recorded and
restored; unexpected buffer changes fail the preflight.

The checks compare zero-gate `H_hat`, normalized deployment RZF, eval output,
and the legacy training-return interface at fixed `rtol=atol=1e-6`. Reports
contain actual maximum absolute/relative differences; the relative-error
denominator floor is `1e-12`, distinct from the unchanged acceptance tolerance.
Power checks sum across users and antennas on every subcarrier.

For each feedback SNR, a separate backward check uses the real task path:
`H_hat -> model.zf_solver -> total-power normalization -> main_train.SumRateLoss`,
with DL noise `10**(-25/10)`. The legacy residual `W_train` is not the loss
input. All populated gradients must be finite; the five gate gradients must
exist, be finite, and have nonzero combined norm. Missing gradients on inactive
FDMA or unused legacy parameters are reported, not incorrectly treated as a
failure. There is no optimizer, parameter update, gradient clipping, scheduler,
or change of precision, determinism, attention backend, noise or RZF settings.

Each actual invocation creates a new timestamp/UUID directory under
`runs_as_screen/diagnostics/as_direct_v1/`, with exclusive, non-overwriting
`report.json`. It records source config/hash/step, code commit/source hashes,
software/effective numerical backends, seeds, errors, power checks, gate
gradients, buffer changes, and pass/fail status. Failed conditions retain their
partial evidence and return a nonzero exit code. Nonfinite report metrics are
represented by null plus explicit field paths and force failure; they cannot
silently prevent a failure report from being serialized. Synthetic rates in
this report are diagnostics, not performance results or checkpoint selection.

Local checks for this tool: **31 standard-library tests passed, 0 failures,
errors or skips**, plus syntax checks and `--dry-run`. The tests cover CLI
isolation, source/config/hash rejection, same-handle hashing, output collision
protection, and valid failure JSON. They use only archived JSON and arbitrary
temporary byte fixtures, not serialized real weights or mocked dependencies.

```bash
python -X utf8 -B -S -m unittest discover -s tests -p test_preflight_proposal_as_direct.py -v
```

**CUDA/real-checkpoint preflight has not been run in this task.** No SSH
connection, checkpoint download, model test rerun, training, performance
evaluation, profiling, plotting, or paper change was performed. Future
training still needs explicit migration before optimizer creation; this tool
does not make the existing `init_checkpoint` branch accept missing gate keys.
