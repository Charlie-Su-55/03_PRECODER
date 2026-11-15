# Fixed-model rate headroom and shared-receiver diagnostics

The completed UE+BS experiment improved both physical arms, but FDMA gained
more than Hybrid. That result calls for a sharper mechanism check before
another architecture or training run. A strong FDMA baseline remains the
control; this batch does not weaken its encoder, budget or training history.

User-specific projections and orthogonal initialization already have close
connections to CDMA and DeepJSCC-PNOMA. This diagnostic introduces neither
mechanism and claims no new projection-based contribution. Changing an image
source to CSI alone does not establish novelty. The immediate research
question is whether the present feedback design has useful rate headroom,
and which receive-error components accompany its remaining loss.

## Fixed scope

| Item | Prespecified value |
|---|---|
| Namespace/output | `twc_v1_k8_rate_headroom_v1`, `runs/twc_v1_k8_rate_headroom_v1/` |
| Source models | Exact completed joint10000 FDMA and Hybrid terminals from `twc_v1_k8_ue_bs_rate_v1` |
| Source evidence | Audited result `ab6bb85254018e463c62e0a15f8a6ff4e7b4330f`, running code `cc3ae7148a87f1333d5e789e59f45398e9c6d3e5` |
| New optimizer updates | Zero; UE/BS/CE updates and power-EMA commits are all zero |
| New data | Namespace above, seed20261126, microbatch4 |
| Sampling | Validation indices8192–8255:64 batches,256 complete MU per arm/gamma |
| Feedback gamma | All0/5/10/15/20/25dB points, using paired channels/noise/pilots |
| Physical protocol | Original K8/Nt32/Nsc256,256RE including64pilot, per-UE Ep16+Ed48 |
| Downlink policy | Original RZF alpha=.03, per-tone total transmit power1, DL25dB |
| Selection | No fitting, checkpoint selection, extra training or test sampling |
| Delivery | Checked scalar CSV/JSON directory; explicit Git publication; no ZIP |

The source checkpoint, receipt, model/statistics identity and source-code
guards must pass before sampling. Source UE/BS/CE weights, CE-error table and
historical power prior remain unchanged throughout scoring. The previous68
scientific core files and all existing run directories remain untouched.

The256 MU are paired across arms and gamma. Six gamma values are six views
of a MU, not six times as many independent channels. Cross-gamma summaries
must average within a MU before sample and batch4 intervals are calculated.
The test role remains sealed. Public feedback gamma and the downlink25dB
parameter are different quantities; no new UE SNR input is added.

## Three rate references

The nominal rate is evaluated with the unchanged decoder output
`H_hat`, unchanged RZF and the simulator's true downlink channel as an
offline scoring label. It is the deployable-model result for this batch.
Three additional quantities help interpret it:

1. **True-CSI RZF reference.** Apply the same alpha=.03 RZF and the same power
   normalization to the true downlink channel, then score on that channel.
   This uses privileged truth only in a separately labelled reference. It
   does not become an input to the nominal receiver or decoder.
2. **True-direction, decoder-norm reference.** For every UE and tone, replace
   the predicted row direction by the true direction while preserving the
   predicted row norm:

   \[
   h_{\mathrm{direction},k}
   =\frac{\|\widehat h_k\|_2}{\|h_k\|_2}h_k.
   \]

   Apply the same RZF and power normalization. This separates a controlled
   direction substitution from replacing the decoder's learned scale as
   well. It is a privileged sensitivity intervention, not a deployable arm.
3. **Interference-free water-filling upper bound.** On each tone, let
   \(g_k=\|h_k\|_2^2\) and use the original downlink noise power \(N_0\).
   Compute

   \[
   U(H)=\max_{p_k\ge0,\;\sum_kp_k\le1}
   \sum_k\log_2(1+g_kp_k/N_0).
   \]

   Discarding interference and applying Cauchy–Schwarz bounds each user's
   desired power by \(g_kp_k\). Maximizing the resulting sum under the same
   per-tone total-power constraint therefore gives a mathematical upper
   bound for the supported linear-precoder rates. It can be loose: the
   genie independently aligns beams and removes their mutual interference.
   Fixed equal power \(p_k=1/K\) is only a feasible point of this genie
   problem, not a general upper bound for unequal-power precoders.

True-CSI RZF is **not** a mathematical upper bound. With fixed alpha and
normalization, a learned `H_hat` can act as a task representation rather than
the most accurate channel estimate. Consequently, true-CSI RZF or the
direction-substitution reference need not beat the nominal model. Retain
signed reference-minus-nominal differences; do not clamp them to zero.

This batch does not run WMMSE. A future true-CSI WMMSE result would also be a
reference/local solution, not a certified global sum-rate upper bound.

## Exact fixed-filter receive-error decomposition

For an exclusive coordinate use its scalar-user receive system; for a
shared coordinate use the actual two-user system. On a selected data RE,
write

\[
y=Gz+n,\qquad \widehat z=Ay.
\]

The normal filter \(A\) is constructed from the estimated UL channel,
historical prior, CE-error table and public noise parameter. Keep that same
filter fixed for the decomposition. True \(G\), transmitted \(z\) and the
realized noise are used only by offline scoring. They never enter a new
nominal recovery filter or decoder.

With \(B=AG\), let \(\operatorname{diag}(B)\) denote the diagonal matrix
retaining only its diagonal entries, and define

\[
e_{\mathrm{self}}=(\operatorname{diag}(B)-I)z,\qquad
e_{\mathrm{cross}}=(B-\operatorname{diag}(B))z,\qquad
e_{\mathrm{noise}}=An.
\]

Then \(\widehat z-z=e_{\mathrm{self}}+e_{\mathrm{cross}}+e_{\mathrm{noise}}\)
sample by sample. Exclusive recovery has no cross-user term. Report all
component powers and the three **signed** cross terms:

\[
\|\widehat z-z\|^2
=\sum_c\|e_c\|^2
+2\Re\{e_{\mathrm{self}}^He_{\mathrm{cross}}
+e_{\mathrm{self}}^He_{\mathrm{noise}}
+e_{\mathrm{cross}}^He_{\mathrm{noise}}\}.
\]

The reported UL scalar is exactly the coordinate mean
\(\operatorname{mean}_{\mathrm{coordinate}}(|e|^2/P_{\mathrm{history}})\),
using the frozen historical power of each selected coordinate. Apply that
same coordinate-wise divisor to every component and signed cross term in
the closure check. This scalar is neither an NMSE formed by dividing total
error energy by actual total transmit energy nor a fraction of downlink
rate loss. Negative cross terms are valid. Component powers generally
do not add to the total MSE by themselves. The self term combines
regularization and own-channel mismatch; it is not a pure CE-error measure.
Likewise, this algebraic decomposition is not a causal attribution of a
downlink rate difference to one component. A frozen decoder's reaction to
privileged inputs cannot be treated as a performance upper bound.

## Decision gates

- If FDMA is close to the useful RZF references, there may be little gain
  available from improving its channel-direction feedback for this chosen
  policy. This does not establish closeness to the information-theoretic
  task limit, especially when the genie bound remains loose.
- If a material direction-reference gap remains, source/decoder task
  representation remains worth examining. Different true-CSI and
  direction-reference responses can expose the role of the learned norm;
  neither response uniquely identifies a cause.
- If Hybrid has much larger cross-user/noise-enhancement components at low
  feedback SNR, effective receive geometry becomes a concrete hypothesis
  for a future sending-subspace study. Large self error instead motivates
  closer examination of regularization and CE mismatch. Signed cross terms
  and downlink outcomes must remain part of that interpretation.
- If both arms have little useful reference headroom, retain that negative
  result. Do not reduce FDMA training, silently simplify its model or alter
  the channel distribution to manufacture a shared-feedback advantage.

This small batch does **not** estimate PCA rank or intrinsic source
dimension. A low single-frame matrix rank would not by itself prove that 24
feedback scalars suffice, because its singular vectors also need a
description. Any later fixed-basis/PCA study must fit on independent
calibration data and score on held-out channels. Retained source energy
alone is insufficient: weak modes may be important for distinguishing DL
users.

## Server commands and evidence

After merging and pulling the diagnostic code on the old server, inspect
the no-weight, no-write plan:

```bash
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B diagnose_twc_v1_k8_rate_headroom.py --plan
```

For CPU checks followed by a detached tmux launch and a timestamped log,
use the checked launcher after pulling `main`:

```bash
bash tools/start_twc_v1_k8_rate_headroom_old.sh
```

Run once with the existing shared-GPU admission policy:

```bash
"$CONDA_PREFIX/bin/python" -B -u diagnose_twc_v1_k8_rate_headroom.py \
  --run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
```

```bash
"$CONDA_PREFIX/bin/python" -B diagnose_twc_v1_k8_rate_headroom.py --summary
```

`--run` requires a fresh diagnostic output root. GPU admission is checked
before creating that root: a busy-GPU rejection is printed to standard
output and does not consume the root, so the same `--run` command can be
repeated when the GPU is available. Once execution has created the root,
any failure preserves its evidence; existing output is not overwritten or
silently resumed. Inspect that failure before choosing a distinct future
attempt. There is no training or `--resume` mode.

When the summary reports a completed export, use the exact report directory
printed there:

```bash
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_rate_headroom.py \
  --publish --export-dir /absolute/report/directory/from/summary
```

Publication is explicit. Only checked scalar tables, fingerprints and
redacted provenance are eligible; no weights, channel tensors, credentials
or private network configuration are uploaded. No ZIP is produced. Keep
the source branch, index, dirty files and original server results intact.

The targeted CPU suite comprises 29 tests: 15 mathematical tests, seven driver
tests and seven publisher tests. These checks cover real small-tensor RZF
calculations, water-filling constraints, exact decomposition and closure,
frozen-state guards, metadata and result handling. Report the actual test
run outcome separately; these checks do not establish real-model performance. Actual
checkpoint loading, CUDA/Sionna sampling and full diagnostic measurements
are executed only by the user on SSH. This diagnostic uses one training
seed and fixed models; its intervals describe validation sampling rather
than training-seed or re-training uncertainty.
