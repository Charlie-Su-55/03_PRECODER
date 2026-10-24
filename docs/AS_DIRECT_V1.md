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

## Validation record and remaining checks

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
The +5 count follows from the five new scalar gates; runtime count verification,
strict synthetic loading, gradients, and numerical replay remain pending the
above CPU command. Real-checkpoint and GPU behavior are also unverified and
are outside this implementation-only task. No performance gain is claimed.
