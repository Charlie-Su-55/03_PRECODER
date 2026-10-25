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
