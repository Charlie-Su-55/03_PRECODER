# FDMA-AS-Lite1024

## Scope

This is a parallel, opt-in architecture. The original model and default training/evaluation behavior are unchanged. A separate Lite entry now reuses the existing trainer through an optional, keyword-only model factory.

| Name | UE FFN | Other architectural changes | Status |
| --- | --- | --- | --- |
| Original | 256 -> 2048 -> 256 | None | Existing AdaptiveHybridPrecoder |
| Lite1024 | 256 -> 1024 -> 256 | None | New AdaptiveHybridPrecoderLite |
| Lite1024-Msg | Lite starting point | Future gated user-level AS message path | Not implemented; no placeholder switch |

Reference: faff8c688802ba88a64ca351155fcd0505c72f0f.
No training, formal evaluation, historical checkpoint loading, or paper modification was performed during code preparation.

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

No MRC/LMMSE, message shortcut or solver change is included. Existing configuration snapshots/fingerprints remain unchanged; only the new Lite-specific configuration adds architecture metadata.

## State loading and metadata

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

LiteTrainConfig.to_dict() saves this metadata plus model_class=AdaptiveHybridPrecoderLite alongside the complete base snapshot. The existing trainer writes that snapshot into config.json and every checkpoint's config; the metadata participates in the Lite fingerprint. The model property itself does not add state_dict extra state. Existing configuration fingerprints and default model loading remain untouched.

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

Actual UE/full-model totals remain unmeasured locally because of the dependency limitation below. The parameter-count test constructs K4/Nt32/Nsc256/Dtot256, four UE layers per branch, five BS layers, D=256, and four subbands WITHOUT forwarding that real-size configuration. The user supplied the following server CPU measurements for commit 873cd6479b546f3293f5fc9936ca6ba96c207515:

| Partition | Original | Lite1024 |
| --- | ---: | ---: |
| encoder | 10,810,274 | 6,607,778 |
| condition | 66,816 | 66,816 |
| ue = encoder + condition | 10,877,090 | 6,674,594 |
| bs_remainder | 9,332,025 | 9,332,025 |
| full | 20,209,115 | 16,006,619 |

The measured reduction is 4,202,496. This partition counts shared cond_embed in ue; it does not imply that condition embedding is used only by UEs. These are user-reported server measurements, not local model construction or performance measurements.

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

CPU model forward/backward/strict-load/parameter-measurement checks are therefore **not executed locally**, not passing numerical tests. The suite explicitly skips this known missing dependency; unexpected import errors still fail. GPU behavior, convergence, performance and historical-checkpoint execution remain unverified. Lite is not connected to any default training entry point.

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

## User-reported server CPU verification

For commit 873cd6479b546f3293f5fc9936ca6ba96c207515, the user reported Python 3.12.13, PyTorch 2.11.0+cu130, working sionna.phy, and all 16 existing model tests passing without skips/errors/failures. This is server CPU functional acceptance, not GPU acceptance or training-performance evidence. The existing model and its 16-test source were not changed for the training entry.

## Independent training entry

Files: train_proposal_lite.py, configs/lite_train_config.py, tests/test_lite_training_entry.py. main_train.train_proposal gains only an optional keyword-only model_factory; None follows the original builder. The factory fully constructs and checks Lite before the original trainer constructs its optimizer. No loop, model, legacy head, loss, RZF, normalization, noise or channel generator is replaced.

The sole formal configuration is K=4, Nt=32, Nsc=256, Dtot=256, Nsb=4, seed=42, allocation=(0,256). It retains full maximum widths (64,256), but the actual training grid is singleton, hence running_pwr has one entry at index 0. Fresh runs start randomly with fresh AdamW/OneCycleLR, with no checkpoint initialization option.

The original k4_n256_standard recipe object is reused, not copied: 50,000 steps, batch 48, LR/weight decay 1e-4, warmup 0.35, gradient clip 1, validation every 500 steps on 32 channels at SNR [0,10,20,25], and mean_all checkpoint selection. Auxiliary weights are (50,100) for the first 20k steps and (5,0) thereafter. FB=DL follows [20,25], [10,25], [0,25] with boundaries 15k/35k. This is not the robust fine-tuning recipe.

Training takes H_hat from the existing training tuple and recomputes normalized RZF(H_hat), not the legacy residual W. Original global-step channel seeds, fixed validation generation and checkpoint selection are reused. Same seed does not imply identical noise/dropout streams to Original: Lite construction consumes extra RNG draws.

Experiment name:

~~~text
hybrid_gnn_specialist_K4_Nsc256_Dfb256_f000_a100_df0_da256_sb4_k4_n256_standard_ueffn1024_seed42
~~~

Paths resolve from the repository location, independent of the shell working directory:

~~~text
runs_as_screen/proposal_lite/checkpoints/<experiment>/best.pth
runs_as_screen/proposal_lite/checkpoints/<experiment>/latest.pth
runs_as_screen/proposal_lite/checkpoints/<experiment>/config.json
runs_as_screen/proposal_lite/checkpoints/<experiment>/status.json
runs_as_screen/proposal_lite/logs/<experiment>/metrics.jsonl
runs_as_screen/proposal_lite/logs/<experiment>/provenance_<UTC timestamp>.json
~~~

Each invocation records its actual runtime Git commit, tracked dirty filenames and SHA-256 of relevant source files. Resume appends a new provenance record, retaining earlier ones. These dynamic records are not put into the config fingerprint and are not claims about old checkpoints.

Only this fixed root is writable. Lexical/resolved paths, symlinks/junctions (including parents and temporary writer paths) are checked. Fresh runs refuse either existing checkpoint or log directories. Resume requires this run's latest/config/status/best/metrics; missing latest never falls back to fresh. Metadata, full config/fingerprint, allocation, progress, optimizer/scheduler/RNG presence and singleton power state are checked before entering the trainer. Completed runs are refused. Resume also requires the same number of visible CUDA devices so all saved CUDA RNG states can be restored.

The unchanged original loader uses load_state_dict's strict=True default: no state filtering or partial loading. It restores model state (including running_pwr), optimizer, scheduler, RNG, step and best metric. Tensor-level strict checks occur inside that trainer, which writes running status before loading; a corrupt tensor state can therefore fail after that status write. A per-experiment exclusive lock prevents concurrent Lite-entry writers. If a process is killed leaving a lock, inspect it and confirm no writer remains before manual recovery; the entry never deletes a stale lock automatically.

## Commands and acceptance order

Run from the repository root after safely pulling this change. No arguments print help; conflicting modes are rejected. The dry-run imports no model/Torch/Sionna and creates no experiment directories. The separate unittest command exercises available CPU dependencies and uses temporary checkpoint fixtures:

~~~bash
python -X utf8 -B train_proposal_lite.py --dry-run
python -X utf8 -B -m unittest discover -s tests -p test_lite_training_entry.py -v
~~~

Next, on a dependency-complete CUDA server only:

~~~bash
python -X utf8 -B train_proposal_lite.py --preflight
~~~

Preflight constructs a separate full-grid real-size model (batch 1), checks pure-FDMA, hybrid and pure-AS CUDA forward/deploy-RZF backward, finite active UE FFN gradients, H/W shapes and per-subcarrier power. It checks default and explicit estimated UL CSI in eval mode. It does not load checkpoints, generate CDL data, take optimizer steps, or write formal files. Allocation indices 0/2/4 belong only to this disposable diagnostic model. Power tolerance is rtol=atol=1e-6; failures report maximum error. No rate ranking or CPU/CUDA bitwise equality is required. Backend settings are reported, not changed to obtain a pass. The existing trainer independently retains its original high float32 matmul precision setting when training is explicitly requested.

Only after GPU preflight passes and the user chooses to start the smoke run:

~~~bash
python -X utf8 -B train_proposal_lite.py --stop-after 500
~~~

This runs steps 1--500, validates and saves best/latest, then pauses. TOTAL_STEPS and OneCycle remain 50,000 with original phase boundaries. Do not repeat the fresh command against an existing run. To continue for another 500 steps, use --resume --stop-after 500; to explicitly finish the remaining schedule, use --resume --train. Standalone --resume is rejected to avoid implicitly launching the full schedule. Do not use --train fresh for the initial smoke check.

The new tests cover original snapshots/fingerprints, unchanged trainer algorithm beyond the factory, factory selection, recipe/metadata, lightweight CLI, output safety and scalar synthetic checkpoint restoration/mismatch rejection. Actual Lite model tests require sionna.phy and explicitly skip when unavailable; scalar fixture tests do not establish actual Lite GPU behavior. GPU preflight and the 500-step smoke run have NOT been executed during this change. FFN512, Lite1024-Msg, formal evaluation, figures and paper edits are not included.

### Local checks actually executed for the entry

On 2026-10-01, the new 32-test suite completed with 30 passing and 2 explicitly skipped, no failures/errors. The skips were the actual Lite CPU factory/checkpoint test (missing sionna.phy) and the real directory-symlink fixture (Windows privilege unavailable). Mocked symlink-decision tests passed but do not replace the real filesystem fixture. The unchanged 16-model-test suite was also rerun: 3 static tests passed and 13 numerical tests skipped for missing sionna.phy. Dry-run/no-argument checks, source AST parsing and diff whitespace checks passed. No optimizer step, GPU preflight, smoke training or real evaluation ran. Pre/post SHA-256 checks confirmed that both model files, the existing 16-test source and pre-existing plotting/figure work were unchanged.
