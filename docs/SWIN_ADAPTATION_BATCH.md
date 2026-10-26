# Swin additional-20k control and immutable ASSpatial archive

This batch prepares the **K4 pure-AS decoder extension** comparison. It does not
replace the original hybrid, multi-user-load, or imperfect-CSI experiments. No
performance improvement is inferred before the new independent test completes.
No paper TeX, model implementation, historical checkpoint, or old result is
modified. The other project is not accessed. There is no cleanup stage.

## Fixed experiments

Only Swin is trained, once for each seed 20261030 and 20261031. Its source is the
original Swin best checkpoint pinned by the archived diagnostic manifest, with
SHA256, saved full config and step checked before `strict=True` loading. No
checkpoint search, weight slicing, or partial loading is allowed.

Each run adds 20,000 optimizer updates: batch48, feedback SNR uniform [0,25] dB,
DL SNR fixed25 dB, task sum-rate only, AdamW weight decay1e-5 and gradient clip0.5.
Encoder/decoder peak learning rates are 2e-5/1e-4, respectively; OneCycle uses
10% warmup and the fixed 20k total. All parameters are trainable. The existing
Swin physical feedback link, normalized RZF and model are unchanged.

The accepted ReplayableChannelData generates the training batches and fixed64
validation channels. Under the same training seed/physical config/batch sizes,
its channel protocol is identical to the proposal screening protocol. Swin
retains the baseline helper's explicitly recorded training-feedback RNG and
SNR draws; identical training noise realizations across unlike architectures
are not claimed. Validation runs at step0 and every500 updates, at FB
0/5/10/15/20/25 dB and fixed DL25. One checkpoint is selected by `mean_all`, the
arithmetic mean of these six rates. Step0 is eligible; negative adaptation is
not hidden by selecting a degraded comparison checkpoint. Selection is never
per-SNR or based on the independent test. Completion of latest at20k is required
even if best remains step0.

The technical first500 are part of the same20k scheduler, not extra updates.
The controller automatically resumes501–20000 after checking the saved state.
Resume rejects missing/full-config/source/fingerprint/group/RNG/scheduler
inconsistencies; it never falls back to source initialization. This is not a
claim that full interrupted training has been verified bitwise equivalent to
uninterrupted training.

`baseline_train.py` has optional factory/data/fixed-DL/step0-selection hooks to
reuse its loop. Legacy calls still use the existing generator, DL=FB, original
optimizer, and selection/saving defaults. New entrypoints opt in explicitly.

## One server batch

New files and responsibilities:

- `configs/swin_adapt_train_config.py`, `train_swin_adaptation.py`: isolated
  config, strict initialization/resume and optimizer groups; no old recipe edit.
- `evaluate_swin_adaptation.py`: six-method shared-channel evaluation, strict
  loads, frozen-state checks and paired sample/batch statistics.
- `as_spatial_archive.py`: validate completed historical evidence and copy
  small artifacts byte-for-byte, with a final immutable archive manifest.
- `run_swin_adaptation_batch.py`: resumable stage orchestration; no old training.
- `swin_adaptation_results.py`: CSV-only six-point plots, comparisons and EN/ZH
  internal drafts generated only after completed new evaluations.

The controller first validates the existing ASSpatial batch selected by
`runs_as_screen/proposal_as_spatial_v1/last_batch.json`: four completed20k runs,
best/latest and sidecar hashes, full configs/fingerprints, completed independent
evaluations, freeze/RNG evidence, CSV coverage and actual parameter statistics.
It does not infer completion from chat or re-run the old experiments.
Historical source hashes are retained as provenance, not compared to today's
newer source as a condition of historical validity.

Small CSVs (all rows/columns), JSONs/manifests and actual parameter records are
copied unchanged into `paper/data/as_spatial_verified/<old-batch>/`. Originals,
weights and raw CSV.gz remain in their original server paths. Those large files
are stream-hashed for identity, never deserialized during archival or copied
into paper/data. Source path/size/SHA256 and historical code provenance are
recorded. Existing archives are reused only if all bytes and their manifest
match; incomplete/unknown destinations are preserved and rejected.

Next: dependency/CUDA/source checks, two Swin first500+resume runs, two unified
evaluations, all-result reporting and a second archive under
`paper/data/k4_swin_adaptation_verified/<new-batch>/`. New training/log/checkpoint
outputs are under `runs_as_screen/swin_adaptation_v1/`; evaluations are under
`runs_as_screen/evaluation/k4_swin_adaptation_v1/`. Completed stages are reused
only after evidence hash verification. A failure preserves its report and
logs and exits nonzero. No rerun of accepted old model/RNG test suites occurs.

## Independent evaluation and interpretation

Test seed20261110, batch8 ×200 =1600 complete multi-user channel realizations
per condition, K4/Nt32/Nsc256/Dtot256/Nsb4, CDL-C, DL25, six FB points0–25.
Each training seed separately compares ASSpatial frozen UE, ASSpatial joint,
Full, ASReceiver, original Swin and its same-seed adapted Swin. All six use the
same actual H_dl/H_ul batches; old means are not spliced into the new CSV.
The existing feedback-RNG protocol is retained. This does not equate absolute
feedback noise power or UE energy between FDMA and AS.

There are36 summary rows per training seed. Signed differences are reported for
Joint−original Swin, Joint−adapted Swin, adapted−original Swin, Joint−Full,
Joint−ASReceiver, and Joint−frozen UE. Six points, low0/5/10 average and all-six
average are retained for each pair. Cross-SNR averaging occurs *within each
sample* before SE/CI computation. Batch intervals use200 means of8 paired
sample differences; sample intervals use1600 paired complete-channel samples.
Both remain available, with ddof1 SE and mean±1.96SE. No seed pooling and no
oracle selection. Preflight, if explicitly requested, uses a separate seed
20261111 and8 samples; one-batch CI is not estimable.

The prior Joint−Full engineering criterion (low mean≥0.5, batch CI lower>0,
FB25 batch CI lower≥−0.1) remains explicitly identified as an engineering
screen, not a publication requirement. The three Swin-focused comparisons are
all reported without post-hoc thresholds, seed selection or deleted low-SNR
points. Intervals describe fixed-checkpoint evaluation sampling, not stability
over training seeds. Only **additional update budgets** match; historical
total training budgets and parameter counts do not. Stored parameter counts
are not MACs or measured latency. Raw representation NMSE is secondary, and
precoder_power is the full-band sum (nominal256), not per-subcarrier1.

## Server commands (not executed on the preparation laptop)

After safe main/origin synchronization, with the accepted `precoder` environment:

```bash
python -B run_swin_adaptation_batch.py --dry-run
mkdir -p terminal_logs
log=$(mktemp terminal_logs/swin_adaptation_batch_XXXXXX.log)
nohup conda run --no-capture-output -n precoder python -B -u run_swin_adaptation_batch.py --run >"$log" 2>&1 < /dev/null &
printf 'PID=%s LOG=%s\n' "$!" "$log"
```

After an interrupted/failed invocation, fix the reported cause and explicitly
use `--run --resume` in place of `--run`; do not start a fresh batch over it.
Resume does not repeat completed, hash-verified training/evaluation stages.

No-download terminal return command, including both seeds, all rates, signed
sample/batch statistics, parameters, selected best steps and scope limitations:

```bash
python -B run_swin_adaptation_batch.py --summary
```

The same command reports failure/stage status if incomplete. Full evidence is
`batch_summary.json`; a verified small tar bundle, vector PDF/PNG previews and
EN/ZH drafts are listed there. paper/data contains only verified CSV/JSON
technical summaries. The controller does not run Git. Review/synchronize only
the two explicit new paper/data archives in a later authorized data commit;
never add runs directories, checkpoints, raw samples or unrelated figures.

## Validation status

Preparation checks actually executed:

- `test_train_swin_adaptation.py`: 16 passed, including construction of real
  AdamW/OneCycle on two CPU leaf tensors (no backward or optimizer update).
- `test_evaluate_swin_adaptation.py`: 14 passed, synthetic rows/metadata only.
- `test_as_spatial_archive.py`: 18 passed, temporary synthetic artifacts only.
- `test_swin_adaptation_batch.py`: 9 passed, orchestration/report contracts.
- AST parsing of all11 Python files and guarded standard-library help/dry-run
  checks for both training/evaluation seeds and the controller passed.

The extra `-S` training-test run passed15 and skipped the optional PyTorch
construction test; the normal installed-PyTorch run then passed all16. No
missing dependency was installed or faked. No real model test is counted in
these57 final passing checks.

Preparation uses only new standard-library/synthetic-data tests, CLI dry-runs
and syntax checks. No real checkpoint is present/read on the laptop; no CUDA
preflight, training, model evaluation, plot generation or server archive has
been run here. Tests check step0 eligibility and strict resume metadata, source
identity, output protection, paired statistics, immutable archival and included
500+19500 orchestration. Real source loading, GPU behavior, 20k completion and
result invariants remain server execution requirements. Do not treat code
preparation or synthetic fixture values as experiment outcomes.
