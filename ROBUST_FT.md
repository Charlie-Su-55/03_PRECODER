# K4 pure-AS：feedback-SNR-robust fine-tuning

本轮只做 Uniform(0,25) feedback SNR + fixed DL=25 的协议对齐实验。
不预设低 SNR 一定超过 Swin，也不预设高 SNR 峰值一定保留。
代码准备阶段未在本地执行 Python、compileall、测试、训练或评估。

## 修改边界与兼容性

- `configs/train_config.py`：新增默认 None 的 `downlink_snr_db` 及只读
  `DOWNLINK_SNR_DB`；新增 `k4_n256_fb_robust_ft`，不加入任何旧 suite。
- `main_train.py`：model 输入仍是 FB SNR；loss noise 只按 DL SNR 计算。
  默认 None 时直接返回原 FB SNR，随机调用次数、validation seed、criterion、
  deploy-consistent RZF 路径和 checkpoint selection 公式不变。
- 旧 recipe 的 `to_dict()` 省略新增的 None 字段，保持旧配置序列化内容及
  `config_fingerprint`，避免旧 checkpoint 续训被错误拒绝。
- 新入口复用 `train_proposal(..., RunOptions(...))`，不复制模型或 loader。
  首次运行调用原 `init_checkpoint` 分支：strict=True，仅加载权重，
  新建 AdamW/OneCycleLR。后续显式 `--resume` 恢复新实验自己的 latest、
  optimizer、scheduler 和 RNG；绝不恢复旧 specialist 的 optimizer。
- 模型、encoder/decoder、UL slicing、noise convention、RZF、baseline、
  specialist 批量入口、原 evaluator、旧 CSV/图和论文 TeX 均不修改。

## 新实验完整训练配置（静态解析）

`--dry-run` 会输出实际解析得到的完整 JSON（含所有模型/物理默认参数和绝对路径）。
主要配置如下；未改变的 baseline 模型兼容字段也包含在 dry-run JSON 中。

```yaml
experiment:
  model_name: proposal
  train_mode: specialist
  k_users: 4
  subcarriers: 256
  feedback_budget: 256
  allocation: [0, 256]
  recipe_name: k4_n256_fb_robust_ft
  seed: 42
  variant: ""
allocations: [[0, 256]]
full_allocation_grid: [[64, 0], [48, 64], [32, 128], [16, 192], [0, 256]]
max_dimensions: {D_F_MAX_PER_USER: 64, D_A_MAX_SHARED: 256}
physical:
  ANTENNAS: 32
  CARRIER_FREQ: 3500000000.0
  SPEED: 1.0
  TOTAL_POWER: 1.0
  NUM_SUBBANDS: 4
proposal_model:
  D_MODEL: 256
  NUM_HEADS: 8
  NUM_ENCODER_LAYERS: 4
  DROPOUT: 0.1
  NUM_UNFOLD_LAYERS: 5
  GNN_AGG_DIM: 48
  NUM_DECODER_LAYERS: 4
  NUM_ITERATIONS: 5
training_recipe:
  name: k4_n256_fb_robust_ft
  total_steps: 20000
  batch_size: 48
  lr: 0.00002
  val_interval: 500
  val_batch: 64
  warmup_pct: 0.10
  grad_clip: 0.5
  phase1_pct: 0.40
  snr_stage1_pct: 0.30
  snr_stage2_pct: 0.70
  stage1_snr: [0.0, 25.0]
  stage2_snr: [0.0, 25.0]
  stage3_snr: [0.0, 25.0]
  val_snr_list: [0, 5, 10, 15, 20, 25]
  downlink_snr_db: 25.0
  dir_mode: real
  ph1_w_dir: 0.0
  ph1_w_mse: 0.0
  ph2_w_dir: 0.0
  ph2_w_mse: 0.0
  weight_decay: 0.00001
  data_cache_size: 1
  log_interval: 100
  best_metric: mean_all
resolved_boundaries: {phase1_steps: 8000, stage1_steps: 6000, stage2_steps: 14000}
optimizer: fresh AdamW
scheduler: fresh OneCycleLR, cosine, total_steps=20000, pct_start=0.10
```

三个阶段都采样 Uniform(0,25)，两阶段的辅助 loss 权重均为零，因此上述边界
不会改变 SNR 分布或启用辅助目标。其他 AdamW/OneCycleLR 默认参数沿用原 trainer。
CDL 设置仍用原 generator：CDL-C、300 ns delay spread、30 kHz 子载波间隔。
Validation 使用原固定信道生成与 feedback-noise seeds；64 个 validation samples
与正式测试的 1600 samples 分离。best 按六点 sum-rate **算术平均**选择。

## 初始化与输出路径

默认初始化（只读，支持显式 `--init-checkpoint PATH`）：

```text
runs_as_screen/proposal_specialists/checkpoints/hybrid_gnn_specialist_K4_Nsc256_Dfb256_f000_a100_df0_da256_sb4_k4_n256_standard_seed42/best.pth
```

新 best：

```text
runs_as_screen/proposal_snr_robust/checkpoints/hybrid_gnn_specialist_K4_Nsc256_Dfb256_f000_a100_df0_da256_sb4_k4_n256_fb_robust_ft_seed42/best.pth
```

同目录另有 `latest.pth`、`config.json`、`status.json`、
`robust_run_manifest.json`；日志在
`runs_as_screen/proposal_snr_robust/logs/<同名实验>/metrics.jsonl`。

新目录已有内容时，不会自动覆盖或重新初始化；必须显式 `--resume`。
如需独立重试，可用 `--output-root runs_as_screen/proposal_snr_robust/retry1`；
只允许专用 robust 根目录或其子目录，禁止写入旧 specialist 根目录。
若首次运行尚未生成 latest 就失败，不应使用 resume；选择新的 retry 子目录，
保留失败现场。对 retry 路径训练、续训和评估必须一致指定 output/robust root。

## 服务器步骤 1：编译、回归检查、dry-run

```bash
cd ~/project/03_PRECODER
python -m compileall -q configs/train_config.py main_train.py train_proposal_snr_robust.py evaluate_proposal_snr_robust.py tests/test_snr_protocol.py
python -m unittest discover -s tests -p test_snr_protocol.py -v
python train_proposal_snr_robust.py --dry-run
```

回归测试只使用配置和假的 model/criterion 检查 SNR 路由，不运行真实网络训练、
channel generation 或 GPU evaluation。dry-run 不加载权重、不启动训练、不创建实验目录。
确认输出中的初始化绝对路径正确、存在，输出属于新 robust 目录。

## 服务器步骤 2：仅运行 500-step smoke，随后停下检查

```bash
python -u train_proposal_snr_robust.py --stop-after 500
```

该命令使用完整 20k schedule，只执行 steps 1–500；不是将 schedule 缩短成 500。
应看到：

- `[Initialize] strict=True`，以及 fresh AdamW/OneCycleLR；
- 日志中 `FB_SNR` 随机变化，`DL_SNR=25.0`；
- validation 列为 0/5/10/15/20/25，fixed DL SNR=25；
- `[Paused]`，状态为 paused，step=500；
- `[Protected source] SHA-256 unchanged`；
- `[Checks passed]` 和位于新目录的 best/latest 路径。

入口会自动核查新 JSONL 中的 SNR 取值/变化、六点 validation 算术平均以及文件存在性。
初始化源 SHA-256 记录在新 manifest，并在返回时重新检查。
这些检查只有在服务器实际运行后才算通过；本地未执行。
**先检查这一步日志，不要直接启动后续长训练。**

## 服务器步骤 3：smoke 通过后续训到总计 20k

```bash
python -u train_proposal_snr_robust.py --resume
```

恢复新 robust 的 latest.pth，从 step 501 到 20000，追加 19500 steps；
保持同一个 optimizer/scheduler，不重新从旧 specialist 初始化。
不要再次带 `--stop-after 500`，也不要在 resume 时带 `--init-checkpoint`。
可先用 `--resume --dry-run` 检查续训路径。

## 服务器步骤 4：固定原协议的五曲线评估

训练完成后：

```bash
python -u evaluate_proposal_snr_robust.py --preflight
python -u evaluate_proposal_snr_robust.py
```

正式配置：FB=[0,5,10,15,20,25]，DL=25，batch=8，num_batches=200，
每点1600 samples，默认 seed=20260921。一次调用原
`baseline_evaluate.evaluate_scenario`，全部方法共享每批同一 H_dl/H_ul。
噪声 convention 和每个条件的 feedback seed 逻辑保持不变；不同架构不保证
逐元素 noise 完全一致。

五条曲线：allocation-conditioned universal 的 (0,256)、original specialist、
new robust specialist、Swin-CFNet FDMA、CsiNet+ FDMA。加载均复用 strict loader；
universal 支持五点但这里只评估 pure-AS，两种 specialist 只支持 pure-AS。
未完成的 robust run 不可作为正式结果。

默认复用 `runs_deployzf_v1` 的 universal 和
`runs_as_screen/baselines` 的 baselines；可显式指定
`--universal-checkpoint`、`--original-checkpoint`、`--robust-checkpoint`、
`--swin-checkpoint`、`--csinet-checkpoint`，不会自动训练缺失模型。

正式输出新目录：

```text
runs_as_screen/evaluation/k4_pure_as_fb_robust/
  summary_k4_pure_as_fb_robust.csv
  samples_k4_pure_as_fb_robust.csv.gz
  plot_k4_pure_as_fb_robust.csv
  historical_curve_check.csv
  sweep_manifest.json
  metadata_k4_pure_as_fb_robust.json
  results_k4_pure_as_fb_robust.json
```

汇总/绘图表各30行，保留 mean、SE、raw NMSE、allocation、method type/key、
seed/sample count。raw NMSE 仍只是 secondary diagnostic。preflight 为每点8样本，
写独立时间戳目录，不可用于论文。已有输出目录一律拒绝覆盖，可显式指定新的
`--output-dir`。

关于历史复现：当前 paper CSV 只存 curve/SNR/mean/SE，没有 seed 或 batch metadata。
默认 20260921 来自当前 screening/envelope 入口。若原始 sweep metadata 不同，
请使用 `--seed 原种子` 并核对其 batch protocol，不能仅凭 CSV 声称 seed 已验证。
正式运行会只读对照当前历史 CSV，输出旧四条曲线24个点的均值差异。
若明显不一致，先核对 checkpoint/seed/protocol，不把两批均值直接拼接。
不会覆盖历史 CSV、图、checkpoint 或论文。
