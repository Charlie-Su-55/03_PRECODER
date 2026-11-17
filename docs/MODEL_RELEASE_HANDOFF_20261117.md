# 两机完成结果与模型交接（2026-11-17）

本记录依据用户回传的两份终态摘要。CSV、逐点区间和来源清单尚待结果分支上传后独立复核；以下数值不代表已经完成云端全报告审计。

## 已完成的结果

旧机完成两条独立 Hybrid UE+BS 轨迹，每条新增10000更新。CE、FDMA冻结；最终六节点等权验证，test封存。

| 指标 / bps/Hz | 低三点0/5/10 | 高三点15/20/25 | 六点平均 |
|---|---:|---:|---:|
| 低SNR续训对照 − Hybrid起点 | +0.198232 | +0.105120 | +0.151676 |
| 加干扰项 − 续训对照 | +0.006137 | +0.000503 | +0.003320 |
| 加干扰项 − 冻结FDMA | −0.870133 | +0.275549 | −0.297292 |

实验轨迹相对FDMA的六点均值差依次为 −2.129663、−0.566954、+0.086218、+0.245876、+0.286031、+0.294739。10dB逐点差的区间尚未出现在终端摘要中；这里只描述均值。显式干扰项的六点平均配对区间为[+0.002100,+0.004540]，实际幅度约为对照速率的0.011%；高三点额外差值区间跨零。

这说明当前权重下的归一化**下行**泄漏惩罚边际作用很小，不能据此认为干扰已经解决。它没有直接训练上行叠加分离。相对起点的增益同时包含额外更新和低SNR采样；Hybrid与冻结FDMA的累计训练预算不同。

新机完成两条独立CE轨迹，每条由CE5000新增60000更新至CE65000。下表是六个训练节点等权汇总，不包含五个中间SNR点。

| 指标 | 原CE5000 | NMSE65000 | geometry65000 |
|---|---:|---:|---:|
| UL NMSE（线性） | 0.04719672 | 0.03180070 | 0.03706585 |
| 方向误差 | 0.04124292 | 0.02913728 | 0.03012284 |
| 用户间Gram误差 | 0.003653126 | 0.001987503 | 0.001588453 |

NMSE续训分别降低上述误差32.62%、29.35%、45.59%。geometry分别降低21.47%、26.96%、56.52%。geometry相对NMSE的Gram误差再低20.08%，但UL NMSE高16.56%、方向误差高3.38%。保留两个末态，不凭目标名称选择模型。尚未验证新CE对实际codec恢复或下行和速率的影响。

原CE预训练已经覆盖连续0–25dB；后续中点检查属于本次六节点续训之外的插值，不能称为全训练历史未见的SNR。

## GitHub交付格式

- 原有 `publish_twc_v1_k8_low_snr.py` 和 `publish_twc_v1_k8_ce_refine.py` 上传标量报告分支，**不含权重**。
- `tools/model_release_03.py` 只按固定科学run清单归档。完整保留checkpoint字节、优化器、EMA、RNG、初始化、收据、校准和验证记录；不裁剪为仅state_dict，也不修改原run。
- `tools/publish_03_model_release.py` 将归档分卷和清单上传为本仓库GitHub Release资产，避免普通Git的单文件100MiB限制。模型不会加入main的文件树。
- 不生成ZIP。模型是未压缩tar流的至多1GiB分卷；每个文件和分卷都有SHA256。报告仍为原CSV/JSON目录。
- old配置覆盖固定清单中实际存在的12个K8科学run，包含历史完整模型和最新两条Hybrid。new配置覆盖CE-refine完整run及其原CE训练源目录。未纳入的目录不会声称已备份。
- 归档清单中的 `source_commit` 是归档工具运行时checkout的提交；真实训练提交仍在原checkpoint/receipt中逐字保留。归档完整性验证不替代原训练程序的模型语义和科学身份验证。
- 重试会查找已有草稿并跳过SHA256一致的已上传资产。若GitHub留下本归档草稿内同名、0字节、无digest的`starter`占位，工具仅清理这个未完成占位后重传；已上传资产内容冲突则停止。
- 仓库是公开仓库，Release资产同样公开；工具不读取或上传代理、Git认证、Overleaf、论文、工作站配置及项目其他目录。

CE的严格加载还检查0到60000每5000的全部13个快照及monitor记录。因此不能只传 `checkpoint_latest.pt`。低SNR模型还依赖其来源链中的CE、原codec、校准器等；old归档保留整个已声明run链。

## 执行顺序

训练已完成后才同步工具代码。两台分别先执行原有报告publisher，回传结果分支和commit，供云端独立审计。随后归档并发布模型。所有操作不启动训练、不连接CUDA、不反序列化checkpoint。

以下命令以本PR已合并main为前提。GitHub连接需要已配置的代理/CA及仓库写认证；训练本身不需要网络。若Release API或上传域名被网关拒绝，本地归档保留，修复网络后重复publish即可，不重复训练。

已激活`precoder`且同步main后，可分别运行 `bash tools/publish_completed_models_20261117.sh old` 或 `bash tools/publish_completed_models_20261117.sh new` 完成以下报告上传、归档和模型发布步骤。脚本不重启训练，已有归档先验证后复用。

旧机：

```bash
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
git pull --ff-only origin main
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_low_snr.py --publish \
  --export-dir "$PWD/runs/twc_v1_k8_low_snr_interference_v1/reports/attempt_63d630146f51"

mkdir -p "$HOME/precoder_model_releases"
"$CONDA_PREFIX/bin/python" -B tools/model_release_03.py --prepare --profile old \
  --output-dir "$HOME/precoder_model_releases/old_20261117"
"$CONDA_PREFIX/bin/python" -B tools/publish_03_model_release.py --publish \
  --bundle-dir "$HOME/precoder_model_releases/old_20261117"
```

新机：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
git pull --ff-only origin main
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_ce_refine.py --publish \
  --export-dir "$PWD/runs/twc_v1_k8_ce_refine_v1/reports/attempt_9e9d46c5e8f7"

mkdir -p "$HOME/precoder_model_releases"
"$CONDA_PREFIX/bin/python" -B tools/model_release_03.py --prepare --profile new \
  --output-dir "$HOME/precoder_model_releases/new_20261117"
"$CONDA_PREFIX/bin/python" -B tools/publish_03_model_release.py --publish \
  --bundle-dir "$HOME/precoder_model_releases/new_20261117"
```

prepare拒绝覆盖已有输出。第一次prepare已成功、只有上传失败时，只重复publish。verify可离线验证已有归档：

```bash
python -B tools/model_release_03.py --verify --bundle-dir /实际归档目录
```

## 新机接管模型

Git pull只同步代码，不能把Release权重自动放回runs。可从GitHub Release下载全部资产到独立目录后verify/restore，也可用一次内网rsync把旧机刚生成的完整归档传到新机。这是文件迁移，不使用旧机GPU、不转发旧机代理，也不改任何IP。

在新机执行，前提是旧机old归档已经完成：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
mkdir -p "$HOME/precoder_model_releases"
rsync -rlt --checksum --partial --info=progress2 \
  -e 'ssh -o ConnectTimeout=10' \
  r50037942@10.204.65.237:/home/r50037942/precoder_model_releases/old_20261117/ \
  "$HOME/precoder_model_releases/old_20261117/"
"$CONDA_PREFIX/bin/python" -B tools/model_release_03.py --verify \
  --bundle-dir "$HOME/precoder_model_releases/old_20261117"
"$CONDA_PREFIX/bin/python" -B tools/model_release_03.py --restore \
  --bundle-dir "$HOME/precoder_model_releases/old_20261117" \
  --destination-repo "$PWD"
```

restore先完整核验归档和目的文件：同路径同SHA文件跳过；同路径不同SHA则在写入任何文件前停止。它不会覆盖新机CE、不删除旧机文件、不修改receipt或降低原模型身份校验。遇到冲突回传报错，不执行reset/clean/强制覆盖。

迁移后，下一批实验代码还须在新机通过原加载器做真实权重严格验收；云端无法代替这项检查。归档工具本身只验证字节和来源记录，不会解pickle或初始化GPU。

## 接下来的科学工作

新机作为训练和综合评估主机；旧机保留完整历史，只做辅助小实验与备份。当前不追加FDMA训练，也不立即继续加长同一Hybrid损失。

先审核两份完整报告的逐SNR曲线、校准和来源，再设计冻结的三CE替换诊断：原CE5000、NMSE65000、geometry65000各配其独立已锁定误差表，使用同一物理输入、UE波形、pilot、噪声、历史功率和RZF。评估实际soft恢复、CSI方向和下行sum-rate，保留六节点和负结果。真UL只作监督或离线评分，不进入可部署接收机。

FDMA如参与该表，只用现有冻结权重前向重评，用于分离CE改变与接入方式改变；不训练FDMA。CE替换会改变decoder输入和可靠度分布，冻结速率降低时要先检查接口失配；不能仅凭CE NMSE决定最终方案，也不能把冻结重评声称为完整适配后的最好性能。
