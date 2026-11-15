# Retired 03 entrypoint cleanup

This cleanup removes five retired entrypoints from the current Git tree. A
repository search found no caller, import, runner or pinned runtime source-SHA
dependency on these paths. They are outside the 61-file ICC/UL-adaptation
scientific core. The old SSH checkout had already reported all five as locally
deleted before the current work.

Historical records are retained. In particular,
`paper/data/k4_pure_as_lite1024/formal_20261002T124602_541062Z/manifest.json`
lists the four `old/` paths in `runtime.tracked_changes`, recording pre-existing
working-tree changes. They are absent from that run's `runtime.source_sha256`;
the change log is not a runtime source dependency. This cleanup leaves the
manifest and those historical entries byte-identical. References inside the
retired scripts themselves also do not make them active dependencies.

The exact original bytes remain in Git at
`895fc71b40baab5cba1744ff457d45f277c46b0f`:

| Retired path | SHA256 of original Git blob bytes |
|---|---|
| `diagnose_cross_cdl_condition.py` | `30b1580d003262c706fc555581e0f00d26b06c7260369b938516d966d02d54d5` |
| `old/diagnostic_eval.py` | `6392dcc72bcf824faafa10e99f6d9263a266761d41084af4296e96239e2642de` |
| `old/train_adaptive_8ue.py` | `6f66737d67f365a2e49bb393da94cc3445997f0917ed83dc1630b62ddb07bfa1` |
| `old/train_csinet_baseline_8ue.py` | `7e6c701be15b043a9c03ac6a1f985bf747517607cd9d97c5a7d956d21ce0f169` |
| `old/train_swin_baseline_8ue.py` | `d83c7e49409e13ac5a3c8bd7cadb57314c46c786a5edfb51867ae349a3b6126b` |

Historical experiment entrypoints that are still imported, called by a runner,
recorded in an identity manifest or needed to interpret a result remain in
place. Existing tests cover those contracts; none was demonstrated to be a
disposable duplicate. The cleanup therefore retains the tests rather than
deleting them by age or by filename. The new joint-training startup runs its
own relevant checks rather than treating every historical test as startup work.

No run directory, checkpoint, training log, result, paper, Overleaf content,
manual figure, Git authentication configuration, proxy profile or 08 project
is a cleanup target. The existing `.gitignore` is unchanged, preserving the
SSH user's unrelated local edits to it.

## SSH after merging and pulling the PR

First stop or wait for this project's actual Python work to finish. A normal
`git pull --ff-only origin main` applies the tracked deletions. Git may refuse
if a retired file has local edits; those edits must be preserved and reviewed,
not reset or overwritten.

The helper is a read-only plan by default:

```bash
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B tools/cleanup_03_obsolete.py
```

Usually every retired path reports `absent`, because Git already removed it.
If an unchanged untracked copy remains, the helper reports
`archive_candidate`. To archive those exact copies:

```bash
"$CONDA_PREFIX/bin/python" -B tools/cleanup_03_obsolete.py --apply
```

Apply requires the official 03 origin, the physical main checkout and no active
or uncertain same-user project Python process. It archives only exact SHA256
matches from the five-path list. Tracked files, staged paths, edited or unknown
copies, symlinks and hardlinks are preserved. Missing files are harmless.

Backups and a per-file receipt live under
`.git/precoder_cleanup/<UTC-and-random-ID>/`. The archive is local to this
checkout and naturally excluded from Git; directories use mode0700 and saved
files mode0600. File bytes are copied, flushed and verified before the original
orphan is removed. Copy or verification failure preserves the original.
The helper verifies that HEAD and index bytes remain unchanged. It never
resets, cleans, stages, commits, pushes, kills a process or deletes a backup.

Restoring a specific archived file is an explicit manual action: inspect the
receipt and copy its saved `files/<relative-path>` to the desired location.
Restoring the original version from the Git commit above is also possible.
There is no automatic restore that might overwrite later work.

Cloud checks use real local Git repositories, filesystem backups and a real
temporary Python process. Actual SSH cleanup is executed only by the user;
no SSH connection or remote cleanup was performed in the cloud.
