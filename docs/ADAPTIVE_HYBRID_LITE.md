# FDMA-AS-Lite1024

## Scope

This is a parallel, opt-in architecture, not a change to the original model or any existing training/evaluation entry point.

| Name | UE FFN | Other architectural changes | Status |
| --- | --- | --- | --- |
| Original | 256 -> 2048 -> 256 | None | Existing AdaptiveHybridPrecoder |
| Lite1024 | 256 -> 1024 -> 256 | None | New AdaptiveHybridPrecoderLite |
| Lite1024-Msg | Lite starting point | Future gated user-level AS message path | Not implemented; no placeholder switch |

Reference: faff8c688802ba88a64ca351155fcd0505c72f0f.
No training, formal evaluation, historical checkpoint loading, or paper modification was performed.

## Construction and initialization

~~~python
from models.adaptive_hybrid_lite import AdaptiveHybridPrecoderLite

# cfg is an existing model-compatible configuration.
model = AdaptiveHybridPrecoderLite(cfg, ue_ffn_dim=1024)
assert model.architecture == "ueffn1024"
assert model.experiment_name == "FDMA-AS-Lite1024"
metadata = model.architecture_metadata
~~~

The subclass completes the original constructor, then replaces only linear1/linear2 in each layer of encoder.fdma_backbone and encoder.aircomp_backbone. Each branch gets one freshly initialized FFN pair, independently deep-copied into its layers. This preserves the original TransformerEncoder template-clone organization: identical initial values within a branch but independent parameter objects/storage. Branch templates are separate. No old weights are sliced. Device/dtype, training flags and requires_grad flags are retained.

Completing original initialization first preserves all non-FFN parameters/buffers for the same initialization seed. The 1024 replacement consumes additional initialization RNG draws; construction does NOT promise the original global RNG endpoint. The 2048 compatibility mode replaces/reinitializes nothing and consumes no additional initialization RNG.

Construct before creating an optimizer. No forward-time replacement, global monkey patch, or copied BS forward is used. Width must be a positive integer, not a boolean.

## Exact architecture boundary

| Module/property | Original | Lite1024 |
| --- | --- | --- |
| Both UE FFNs / all four layers per branch | 256 -> 2048 -> 256 | 256 -> 1024 -> 256 |
| UE attention | D=256, eight heads | Unchanged |
| UE activation / norm order / dropout | ReLU / post-norm / cfg.DROPOUT | Unchanged |
| Input/output projections, condition/position embeddings, pooling | Original | Unchanged |
| Per-subband prefix selection / maximum output widths | Original | Unchanged |
| Joint power normalization / running_pwr | Original | Unchanged |
| Physical feedback / exact UL slicing / noise convention | Original | Unchanged |
| BS cross-subband FFN | 256 -> 2048 -> 256 | Unchanged |
| BS refinement FFN | 256 -> 512 -> 256, GELU | Unchanged |
| GNN / attention / representation / legacy residual heads | Original | Unchanged |
| RZF / forward return interface / H_ul_est | Original | Inherited unchanged |
| Allocation support | Supplied cfg's allocations | Unchanged |

No MRC/LMMSE, message shortcut, solver change, configuration change or fingerprint change is included.

## State loading and future metadata

State-dict keys correspond exactly to the original. For 1024, shape changes are limited to UE linear1.weight, linear1.bias, and linear2.weight. UE linear2.bias keeps width 256 but is also freshly initialized. The 2048 compatibility mode has original shapes.

Always load strictly. Loading 2048 state into 1024 or vice versa must raise size mismatch. There is no slicing, filtering or partial-load helper, and no real checkpoint was used.

The JSON-serializable architecture_metadata property reports:

~~~json
{
  "architecture": "ueffn1024",
  "ue_ffn_dim": 1024,
  "ue_layers": {"fdma": 4, "as": 4},
  "D_MODEL": 256
}
~~~

A future Lite-specific training entry MUST save this alongside its full model/configuration metadata, reconstruct the appropriate class/width, and load strictly. This property does not save checkpoints or add state_dict extra state. Existing checkpoint formats, default loaders and configuration fingerprints are untouched.

## Parameter accounting

Two biased FFN linear layers contain 2*D*F + F + D parameters. For two UE branches with L layers each:

~~~text
reduction = 2 * L * (2048 - F) * (2*D + 1)
D=256, L=4, F=1024: 4,202,496 parameters
~~~

The following are formula-derived counts, NOT measurements from a locally constructed model:

| Eight UE FFNs combined | Original | Lite1024 | Reduction |
| --- | ---: | ---: | ---: |
| Parameters | 8,407,040 | 4,204,544 | 4,202,496 |

Actual UE/full-model totals remain unmeasured locally because of the dependency limitation below. The parameter-count test constructs K4/Nt32/Nsc256/Dtot256, four UE layers per branch, five BS layers, D=256, and four subbands WITHOUT forwarding that real-size configuration. It reports encoder parameters, shared cond_embed parameters, their sum (ue), the remainder (bs), and the full model. This accounting partition does not imply that cond_embed is used only by UEs; it is shared with BS processing.

## CPU-only verification

Run from the repository root in an existing dependency-complete environment:

~~~bash
python -X utf8 -B -m unittest discover -s tests -p test_adaptive_hybrid_lite.py -v
~~~

The tests use small synthetic CPU channels, not generated CDL data or real checkpoints. They temporarily use one CPU thread and restore the thread count and CPU RNG. They do not change precision, deterministic-algorithm settings, attention backends, or noise generation. No optimizer steps or training loops are run.

Coverage includes UE/BS widths; exact parameter reduction; independently cloned FFNs; same-seed non-FFN state; exact 2048 state/RNG and train/eval replay; endpoint/hybrid shapes/power; default/explicit estimated UL CSI; strict synthetic round trips and size-mismatch failures; and active-UE-FFN gradients.

The backward test takes H_hat from the training tuple, recomputes RZF(H_hat), normalizes per-subcarrier power, and constructs the current sum-rate loss. It does NOT use the legacy residual W as the task target, nor demand gradients from unused branches/residual heads. Replay requires exact equality and reports condition/max error on failure. Physical power checks use a stated 2e-6 absolute tolerance.

### Local execution boundary (2026-10-01)

Python 3.12 / PyTorch 2.6.0+cpu is installed. However, the original model import executes utils/__init__.py, whose channel-generator import requires the missing sionna.phy module. The other installed Python (3.11) has neither Torch nor Sionna. No environment was installed, old utility changed, or import stub/monkey patch used to bypass this dependency. UTF-8 console mode only avoids an existing Windows warning-print encoding error.

CPU forward/backward/strict-load/parameter-measurement checks are therefore **not executed locally**, not passing numerical tests. The suite explicitly skips this known missing dependency; unexpected import errors still fail. GPU behavior, convergence, performance and historical-checkpoint execution remain unverified. Lite is not connected to any default training entry point.

The exact command above was executed locally: 16 tests discovered, 3 static/source-integrity checks passed, and 13 CPU-model checks explicitly skipped; exit code 0. The passing checks cover the canonical original-source hash, the parameter-reduction formula, and the subclass execution-path/replacement boundary. Both new Python files also passed AST parsing. A zero suite exit code with these skips is NOT numerical validation of Lite.

## Original source integrity

Starting Windows raw-file SHA-256 (also checked after edits):

~~~text
EFAAD80D29D238326BCDD8A7E8E4212705A00D7A2EDBA1BFF8BFB565A836AD25
~~~

For the portable unit test, CRLF-normalized-to-LF SHA-256 is:

~~~text
67dd73c7720a2aa7c4e33b0f5dbaa30e7824c3aa538fba6f5f73739ecd951ceb
~~~

The raw local hash is checked separately. LF normalization prevents a false source-integrity failure on a Linux Git checkout.

Next action in a compatible environment: run the CPU-only command above and record the actual totals/results. Do not start training as part of this verification. FFN512 and Lite1024-Msg experiments are not implemented or initiated by this change.
