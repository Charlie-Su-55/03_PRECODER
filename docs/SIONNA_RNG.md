# Standalone Sionna channel RNG isolation, v1

## Evidence and scope

The user reported the following server observations (not independently rerun on
the laptop): Sionna **2.0.1** starts with `config.seed=None`; resetting its seed
explicitly made repeated `H_dl`/`H_ul` hashes identical. Inspection of the current
`main_train.py` confirms that its seed/capture/restore helpers cover default
Python, NumPy and Torch CPU/CUDA RNGs, but not Sionna's separate generators.

The pinned [official Sionna 2.0.1 Config implementation](https://github.com/NVlabs/sionna/blob/v2.0.1/src/sionna/phy/config.py)
uses separate Python, NumPy and per-device Torch generators. Its seed setter
replaces those generators and also reseeds default Torch CPU/CUDA generators,
including when Config is first initialized. Restoring the old seed alone
therefore does not restore advanced random streams.

This change is an **independent diagnostic and in-process isolation utility**.
It does not fix or change existing training/evaluation protocols, reinterpret
old result files, or validate full training resume. No old checkpoint,
configuration fingerprint, model, channel formula or paper file is changed.

## Isolation API and supported runtime

`utils/sionna_rng.py` provides:

- `isolated_sionna_seed(seed)`: an explicit integer seed in `[0, 2**32-1]`,
  excluding bool/None. Nested contexts are supported; the yielded value is the
  genuine Sionna Config object.
- `snapshot_rng_state()`: an in-process snapshot with `restore()` and a
  JSON-compatible `fingerprint()` for diagnostic comparisons.
- `runtime()`: lazy real-dependency loading, with strict Sionna **2.0.1** version
  and private RNG-layout checks. Unsupported versions/environments fail; no
  alternate implementation or fake dependency is substituted.

Each context saves the actual advanced states, not just initialization seeds:

| State | Saved and restored |
| --- | --- |
| Sionna | Seed including None; original Python/NumPy generator objects and states; lazy None pools; original Torch registry and generator objects plus each device's state |
| Defaults | Python random state, legacy NumPy global state, Torch CPU and all visible CUDA RNG states |
| Config | Device and precision values, without changing the numerical backend |

On entry, Python/NumPy defaults and Sionna are explicitly seeded. Sionna retains
its official per-device offsets for both explicit and default Torch generators.
On normal, exceptional or nested exit, original objects and exact states are
restored directly. The exit path does **not** invoke the seed setter. The
version-specific private access is confined to `_Sionna201State`; default RNG
capture/restore also surrounds the first Sionna import.

This is a process-global, cooperating single-threaded API. Its reentrant lock
serializes its own contexts, not unrelated background users of global RNGs.
Changing visible CUDA devices during a context is unsupported. CUDA context
initialization may occur to capture all CUDA RNG states; initialization itself
cannot be undone. Snapshots are not serialized checkpoint objects and must not
be treated as a cross-process training-state restoration format.

For a private Python/NumPy RNG that was still lazy None under seed=None, the
restored state is precisely "not yet initialized". Its future entropy-based
first initialization is not a pre-existing stream that can be predicted. The
diagnostic checks lazy-state restoration separately, then checks next draws
for materialized, already-advanced private streams.

The existing `utils/__init__.py` imports the channel module and calls GPU setup.
The standalone entry therefore loads this utility and the unchanged channel
module explicitly by their file paths, avoiding that package-init side effect.
It does not replace imports globally or patch Sionna. Existing application code
that imports the `utils` package still has its original initialization behavior.

## Channel-only CUDA acceptance

`diagnose_sionna_rng.py` defaults to help. `--dry-run` is standard-library-only,
does not read weights or initialize a GPU, and creates no output directory.
`--run` requires genuine Sionna 2.0.1 and CUDA; there is no CPU fallback.

The test calls the original `CDLChannelGenerator_Feedback` with K4/Nt32/Nsc256,
CDL-C, B=2 and unchanged defaults. Its `generate_batch_data` produces downlink
then uplink in two consecutive draws. There is only one seed context around
the complete pair; **no reset between DL and UL**. No checkpoint, model,
training update, rate evaluation or profiling is involved.

Acceptance covers fixed-seed replay using a reused and a rebuilt generator,
distinct seeds, validation interleaving, reordered explicit-seed requests,
a fresh subprocess, and preservation of external RNG state/next draws under
normal, exceptional and nested exits. Seed=None and lazy/populated private RNG
states are included. DL and UL hashes, maximum numerical discrepancies,
dependency/backend details and checks are recorded. Backend settings are
reported, not changed to manufacture agreement.

Replay acceptance requires identical DL/UL byte hashes as well as a numerical
comparison at fixed rtol=atol=1e-6. The actual maximum absolute/relative errors
are retained. A small discrepancy does not silently pass if hashes differ.
Fresh-process arrays travel only through subprocess IPC; no channel arrays are
saved to disk. The installed Sionna Config source hash is also recorded.

Every real invocation creates a fresh directory under
`runs_as_screen/diagnostics/sionna_rng_v1/`, with only report/log evidence.
Existing outputs are never overwritten. Any required failure produces a
nonzero exit and retains a failure report. Passing establishes channel replay
in the tested runtime, not exact optimizer/model/dropout/scheduler resume or
cross-hardware reproducibility.

## Future protocol recommendation (not integrated)

A future explicitly versioned channel protocol should derive seeds from a
stable mapping of `(protocol_version, base_seed, split, global_step, batch_id)`.
Use a documented cryptographic digest/encoding, not Python's randomized `hash`.
Training, validation and held-out test splits should have separate namespaces;
for fixed validation data, use a fixed validation index instead of the current
training step. Seed the **whole DL/UL pair** once, preserve both consecutive
draws, and distinguish constructor-time from sampling-time randomness.

Record the protocol version, seed mapping, Sionna version, physical config and
source hashes in any future run. Test resumed versus uninterrupted execution
separately, including all other RNG users and numerical/backend behavior.
Do not silently apply this protocol to historical checkpoint continuation or
upgrade old config fingerprints/results. That would be a new experimental
protocol, not a backwards-compatible reinterpretation of old results.

## Local and server validation

Local installed metadata is Torch 2.6.0, NumPy 1.26.1, **Sionna 0.15.1**. The
required 2.0.1 runtime is therefore unavailable; no dependencies are installed
or mocked. True Sionna/CUDA checks must be reported as skipped/unexecuted here.

Actual local checks: **23 tests discovered: 15 passed, 8 explicitly skipped,
0 failures/errors**. The skips are the real Sionna RNG tests, including a
fresh-process first-import test for preserving existing default streams.
The executed tests cover seed/type/range rejection, real unsupported-version
rejection before heavy imports, read-only help/dry-run, exclusive output paths,
failure-report persistence/nonfinite evidence, and the unchanged consecutive
DL/UL sampling call. AST syntax checks of all three Python files and a separate
`python -B -S diagnose_sionna_rng.py --dry-run` passed. No model, channel,
CUDA acceptance, SSH connection or training was run locally.

Server commands (not executed during implementation):

```bash
python -X utf8 -B -m unittest discover -s tests -p test_sionna_rng.py -v
python -B -S diagnose_sionna_rng.py --dry-run
python -B -u diagnose_sionna_rng.py --run
```

The test file contains only lightweight metadata/CLI/output checks and small
dependency-gated RNG tests; actual CDL replay is exercised by the explicit
CUDA diagnostic command. Neither command trains or evaluates a model.

## Optional trainer data interface (new, default disabled)

### Prior server acceptance, supplied by the user

The user reported **22 RNG utility tests passed, 1 expected skip**, and **all
17 channel diagnostic checks passed**, including fresh-process replay. Report:
`runs_as_screen/diagnostics/sionna_rng_v1/channels_20261025T192137_887475Z_0774b71b/report.json`.
This is user-provided server evidence. The report was not read or rerun locally
in this implementation task, and the accepted RNG utility is unchanged.

### Minimal opt-in boundary

`main_train.train_proposal(..., channel_data=None)` keeps the existing channel
constructor, validation-set helper and per-step training helper when the
argument is None. The legacy helpers' unconditional cross-process/resume claims
were corrected in comments; their RNG behavior and code paths are unchanged.
The original model, optimizer, scheduler, losses, feedback noise, cache policy
and update loop are not replaced. No old recipe or training CLI opts in.

The new `utils/replayable_channel_data.py` exports a metadata-only constructor:
`ReplayableChannelData(base_seed=..., k_users=4, nt=32, nsc=256, batch_size=...,
validation_batch_size=..., carrier_freq=3.5e9, speed=1.0, cdl_model="C",
device="cuda:0")`. Its interface is:

- `protocol`: a copy-safe JSON description including version, data identity,
  seed mapping, physical settings, batch sizes and execution device/precision.
- `prepare()`: construct the original generator inside the accepted RNG
  isolation context. This is the trainer's alternative to its old constructor.
- `validation()`: regenerate the same fixed validation pair, independent of
  training step or call order.
- `training(global_step)`: generate the pair addressed by a 1-based update step.

Each sampling call uses one isolated seed around the original
`generate_batch_data`: DL followed by UL, with no reset in between. Constructor
and sampling isolate external RNGs. The generator's constants, normalization
and operations are not edited. Nt remains 32 because its physical BS array is
fixed at 4x8. The object returns the generated device tensors directly; it does
not hash or copy every training batch to CPU. Only RNG-state capture performed
by the already accepted isolation utility remains. No Sionna RNG object or
channel tensor is added to checkpoint payloads.

### Exact seed mapping and data identity

Protocol: `training-channel-data-v1`; mapping: `namespace-offset30-v1`.
The canonical identity is sorted, compact, ASCII-escaped JSON encoded as UTF-8.
Its SHA-256 hex digest is `data_id`. Identity includes the base seed, protocol
and Sionna/isolation versions, physical settings, both batch sizes, and declared
execution device/precision. No model/arm name, filesystem path, timestamp or
Python `hash()` enters the identity or seed derivation.

For each split, take SHA-256 of UTF-8
`protocol_version + ":" + data_id + ":" + split`; interpret its first four bytes
as an unsigned big-endian integer and reduce modulo `2**30` to obtain `offset`.
The seed is `(namespace << 30) | ((offset + index) % 2**30)`:

| Split | Namespace | Index |
| --- | --- | --- |
| constructor | 0 | 0 |
| validation | 1 | 0, never the training step |
| training | 2 | global_step, 1 through 2**30-1 |

Base seeds must be actual integers in `[0, 2**32-1]`; bool/None are rejected.
Derived seed bounds, the namespace registry, supported version and complete
serialized mapping are checked. Namespace ranges are disjoint, and the
offset-plus-index mapping has no collisions between supported training steps
within one data identity. Finite 32-bit seeds do not imply collision freedom
between arbitrary different identities. Replay is promised only for the same
declared configuration and supported runtime, not identical per-sample data
after changing batch size, batching boundaries, precision or hardware/backend.

The trainer's existing prefetch cache simply calls `training(future_step)`.
Fetching a different step first or inserting validation calls therefore does
not change the address of any other training batch. The diagnostic tests this
property directly, not by pretending to resume a model checkpoint.

### Explicit config and strict new-protocol resume

A **new** config must explicitly include the exact `data.protocol` value under
`channel_data_protocol` in `to_dict()`. The existing fingerprint algorithm
already hashes all of `to_dict()`, so this opt-in declaration participates in
the fingerprint without changing the algorithm or old serialized configs.
It is not enough to attach an un-serialized Python attribute to an old config.
This task does not provide a new training recipe/entry or modify TrainConfig,
ASDirect config, or their fingerprints.

Before any trainer output directory/status/config creation, validation rejects:

- An object without a declaration, a declaration without an object, or an
  object whose protocol differs from the declaration.
- Unsupported/mutated protocol metadata, inconsistent physical dimensions or
  training/validation batch sizes, and an unsupported step range.
- For the new protocol, an existing config snapshot that differs in either
  full serialized content or fingerprint.
- Explicit new-protocol resume without latest, missing full saved config or
  fingerprint, any full-config/fingerprint mismatch, and incomplete resume
  payload fields. An old checkpoint cannot silently become a new-protocol run.

New-protocol resume reads and validates the CPU payload before output creation,
then passes that same payload to the existing loader, which repeats the strict
metadata check before loading state. It is not reread from a potentially
different file. Fresh runs must explicitly use resume=False and new run paths;
an existing latest with resume disabled is rejected. Historical missing-
fingerprint behavior remains confined to the old, undeclared protocol. Explicit
weights-only initialization is distinct from resume and is not a protocol
upgrade of a historical run. No such training is started here.

### Narrow data-interface diagnostic and actual checks

`diagnose_training_channel_data.py` uses the same declaration validation,
`prepare`, `validation`, and `training(global_step)` operations as the trainer.
Its independent diagnostic config uses CUDA, B=2/K4/Nt32/Nsc256, CDL-C; it never
imports the trainer, constructs a model/optimizer, or reads a checkpoint.

It compares selected steps 1,2,5,500,501,5000 in ascending order, interleaved
with fixed validation, and in reversed prefetch order. A fresh subprocess first
requests step501, then5000 without generating earlier steps. It compares DL/UL
hashes and actual max absolute/relative errors, along with validation hashes.
Full external default/Sionna RNG fingerprints are checked around construction,
preparation and each data call. Failed side effects are reported before any
cleanup restoration. This does not repeat the previously accepted complete
normal/exception/nested RNG isolation suite.

Reports go to a new exclusive directory under
`runs_as_screen/diagnostics/training_channel_data_v1/`; protocol, dependency and
source versions/hashes, backend settings and selected channel evidence are
saved. Failed checks preserve a report and return a nonzero exit code.

Actual local validation of this interface: **42 tests: 41 passed, 1 explicitly
skipped, 0 failures/errors**. The skip records the unavailable genuine Sionna
2.0.1/CUDA acceptance environment (local Sionna is 0.15.1). Tests execute pure
metadata helpers and AST-extracted actual trainer guard functions, covering
the old None branch, rejection before output/runtime work, mapping, protocol
binding and strict resume. They do not train a stub model or fake Sionna.
The existing ASDirect entry's **41 standard-library regression tests passed**.
Syntax and standard-library help/dry-run checks passed. No dependencies were
installed, no weights downloaded, and no training/model/channel evaluation ran.

Server interface acceptance commands, not executed locally:

```bash
python -B -m unittest discover -s tests -p test_replayable_channel_data.py -v
python -B -S diagnose_training_channel_data.py --dry-run
python -B -u diagnose_training_channel_data.py --run
```

**Boundary:** this validates channel reproducibility through the new data
interface only. Full model/optimizer/scheduler pause/resume has not been shown
bitwise identical; other random streams, numerical kernels, and trainer
initialization still require a separate controlled test. Historical checkpoints
and results are not silently reclassified under this new protocol.
