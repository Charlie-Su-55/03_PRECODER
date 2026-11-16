# 独立多SNR CE续训与几何辅助实验

本批可以在新SSH独立运行，不等待旧机正在执行的low-SNR Hybrid训练。它只读取早期已完成的CE5000，训练pilot-only CE U-Net。UE编码器、BS codec decoder、FDMA和历史power-EMA的新增更新均为0，也不改旧机当前进程。

## 本批解决什么问题

此前CE仅训练5000次有效更新；它已经有公共反馈SNR输入，但尚没有完整逐SNR验证学习曲线证明该预算充分。已有早期机制诊断显示CE误差会通过共享接收滤波器放大，且仅改善CE/soft误差并不自动改善下行sum-rate。

本批用两条同源、同数据、同预算的CE轨迹区分“继续训练”与“改变CE监督方向”：

| 项目 | 固定配置 |
|---|---|
| Namespace / 输出 | `twc_v1_k8_ce_refine_v1` / `runs/twc_v1_k8_ce_refine_v1/` |
| 初始化 | 原ICC-aligned CE5000完整模型，严格核验checkpoint、receipt、源码核心和模型状态 |
| `nmse`轨迹 | 仅物理UL NMSE；新增60000次CE更新 |
| `geometry`轨迹 | UL NMSE + 空间方向误差 + 用户间归一化复Gram误差；后两项权重均固定为1；新增60000次CE更新 |
| 每条末态 | 原5000 + 新60000 = 65000次CE更新；两条从同一起点独立分叉，不串联 |
| 优化器 | 每条fresh AdamW，LR从1e-4按cosine降至1e-5，weight decay1e-4，clip1 |
| batch | microbatch4 × 累积8 = 32个完整8UE MU / 有效更新 |
| 训练SNR | 公共反馈gamma = 0/5/10/15/20/25 dB，逐MU均衡轮转；每个三更新周期完全均衡 |
| CE网络与物理 | 现有pilot-only residual U-Net；rank8 Fourier-LS初值，ridge.001；K8/Nt32/Nsc256，64pilot，Ep16/UE，原N0定义 |
| CE输入 | 已知pilot、pilot观测、LS初值、mask及 `(gamma-12.5)/12.5`；不引入真UL作为网络输入 |
| 保存与曲线 | 每100更新保存；起点和每5000更新保留模型快照与逐SNR曲线；首100包含在60000预算内 |
| 模型选择 | 固定60000新增更新末态，不用曲线选best checkpoint |
| test | 继续封存 |

两条末态都为65000次；不是把两条串起来训练到125000次。原CE5000曾在连续Uniform[0,25]预训练，因此中点只表示本次六节点续训未采样的位置，不能声称是全历史未见SNR。

CE训练只发送Ep16的pilot，不发送Ed48的codec数据；噪声仍按原协议 `N0=.25×10^(−gamma/10)`，其中.25来自原系统参考64/256，不改成pilot能量重新定义gamma。下文48/44只用于离线shared恢复分析。

计数：每轨新增1,920,000个完整MU、15,360,000条UE链路输入；两轨合计120000次新CE更新。这个预算用于同时检查长训练收敛与几何辅助，不等于更长训练必然改善。

按旧机CE约0.139秒/更新估计，120000次约4小时38分钟，再加校准/验证和保存。新机速度、驱动和负载可能不同；实际ETA在启动后量测，**不是固定4～5小时的硬保证**，也不会为填满时间自动延长预算。

## 损失与独立诊断

令真实UL为H、估计为Hhat。

- UL NMSE：每UE复误差能量除真实链路能量，再在UE/MU上平均，与原CE训练的物理目标一致。
- 空间方向误差：逐UE、逐tone的 `1-|hhat^H h|²/(||hhat||² ||h||²)`，对相位和幅度尺度不敏感。
- 用户间Gram误差：逐tone将每UE天线向量归一，比较8×8复Gram的非对角项，再平均。这是几何辅助目标，不是已证明最优的目标。

两条轨迹共享实际信道、pilot polarity、标准噪声和gamma序列，均使用相同优化器与独立模型存储。A/B比较检验几何辅助目标；相对原CE5000的比较同时包含新增训练、SNR采样和学习率配方变化，不能全部归因于某一个因素。真UL只作监督/离线评分标签；训练没有涉及UE发送码字，也不需要最新Hybrid权重。

最终还计算一个**均匀符号prior下的shared解析恢复误差**：对G4每组32×2估计信道设计原风格正则化滤波器，prior固定48/44，CE方差来自独立校准；真实UL只进入误差评分。它帮助判断CE改善是否有利于线性分离，**不是实测codec NMSE，也不是下行sum-rate**。

## 校准、验证与导出

先严格加载原CE5000及两条CE65000末态，再生成最终验证数据。三个模型在评价时全部冻结，原始源文件和状态在前后均核验。

- 校准：新namespace的独立`ce_calibration`角色，六gamma各64批×4 =256 MU；分别生成三模型的 `[6,8,256]` CE绝对误差方差表并锁定。
- 曲线：固定monitor数据，六gamma各16批×4 =64 MU，起点与每5000更新记录；不用于选末态。
- 六节点最终验证：新的validation4096–4351，各1024 MU / gamma / 模型。
- 续训节点外的插值SNR点：2.5/7.5/12.5/17.5/22.5 dB，独立validation8192–8447，各1024 MU / gamma / 模型。
- 同一MU跨gamma重复观察，先在MU内平均，再按4 MU物理批次计算抽样区间；不是把gamma当独立训练seed。
- 原始完整MU标量共33792行，全部节点及负结果保留。该结果仍条件于一个源训练seed和固定校准表，不证明多seed泛化。

自动生成标量CSV/JSON目录，不生成ZIP：

```
README.md
samples.csv.gz
summary.csv
comparisons.csv
input_fingerprints.json
provenance.json
calibration.json
curve_summary.csv
export_manifest.json
```

`calibration.json`是锁定的误差统计，不是模型权重。公开报告不含channel tensor、checkpoint、密码、机器私有路径。上传必须显式调用publisher，训练不会自动上传。

## 合并PR后，在新机一键启动

目标是新机`rong@admin1-ELITE`，仓库仍为`/home/r50037942/project/03_PRECODER`，Conda环境为`precoder`。新机此前迁移的早期run应已包含源CE5000，因此不需要等旧机本轮结束。

在新机粘贴：

```bash
bash <<'START_CE_NEW' || printf '%s\n' '同步或启动停止，请回传上面的错误。'
set -Ee -o pipefail
test "$(id -un)" = rong
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER

test "$(git branch --show-current)" = main
git status --short --branch
git pull --ff-only origin main
bash tools/start_twc_v1_k8_ce_refine_new.sh
START_CE_NEW
```

launcher先检查已声明科学核心是否存在本地改动，以及原CE文件、现有输出/tmux会话和CPU测试，再以`k8_ce_refine` tmux会话启动新任务。未涉及科学核心的overleaf、.gitignore等本地修改会保留。它不重置或清理工作区，不安装包，不接触旧机，不停止任何进程。旧机当前训练结束前保持其checkout不变，不为本批在旧机执行git pull。已有输出时会保留并停止，不能把旧的aligned目录当作续训输出。

启动参数沿用：

```
--run --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
```

源文件要求：

```
runs/twc_v1_k8_icc_aligned_savefix_v1/training/ce/checkpoint_latest.pt
runs/twc_v1_k8_icc_aligned_savefix_v1/training/ce/terminal_receipt.json
runs/twc_v1_k8_icc_aligned_savefix_v1/training/ce/manifest.json
runs/twc_v1_k8_icc_aligned_savefix_v1/training/ce/initialization.json
```

源checkpoint SHA256为 `95c1e786b2036ecab81a865408555069ae3ce699620d1afacd14fc11982db494`。
所有文件和实际模型状态都需要严格加载核验，文件存在检查不代替实际验收。

本机训练不需要GitHub或HTTP代理持续在线；拉代码与上传报告时需要可用网络。GPU预检只是启动时检查，不是硬资源隔离。

## 查看进度与回传

在新机另一个终端执行：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER

date -u
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_ce_refine.py --summary
```

summary读取本地已保存边界，显示两条CE各自新增更新、累计更新、checkpoint时间与量测ETA。顶部阶段时间不一定等于最新checkpoint时间。

需要看日志时，使用启动命令打印的准确Log路径，例如`tail -n 60 该路径`。进入任务：

```bash
tmux attach -t k8_ce_refine
```

离开tmux而保持任务运行：按`Ctrl-b`，松开，再按`d`。

确认summary为`completed`后回传：

```bash
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_ce_refine.py --verify-only
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_ce_refine.py --publish
```

将输出中的Read URL、Result commit和summary发回即可。结果分支只供读取，不需要合并到main；publisher保留当前HEAD、index及未提交文件。上传失败时本地结果和准备好的commit保留，可以修复网络/认证后重试相同命令。

## 中断后的显式恢复

先查看summary和tmux，确认原任务没有继续运行。不要删锁文件、删输出目录或重启已存在的会话。需要恢复时可在一个新的tmux交互会话内运行：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
"$CONDA_PREFIX/bin/python" -B -u run_twc_v1_k8_ce_refine.py \
  --run --resume --allow-shared-gpu --min-free-mib 12000 --max-gpu-util 80
```

它只恢复本namespace的完整保存边界，不续训旧aligned批次，不放宽来源校验。

## GitHub报告与checkpoint是两种交付

旧机上传的结果分支让云端和新机读取标量报告，**不会传递真实模型权重**。本批读取的是新机已迁移的原CE5000，所以能独立开始。之后如果要用新CE继续Hybrid训练，需要把选定CE的完整run/checkpoint及收据在两机间通过内网rsync或SFTP同步，并再次核验SHA与来源；不能把拉到结果分支当成已获得模型。

新CE不会自动替换旧机正在执行的CE或原误差表。后续把它接入Hybrid时，要明确选择的候选、误差表和BS适配预算，再使用新namespace。

## 云端验证边界

云端只执行源码检查、真实CPU小张量梯度/优化器/序列化/恢复测试、统计重算与本地bare Git测试。真实CUDA、Sionna、原CE权重加载、4～5小时耗时和实际CE/下行性能只能在SSH上验证。轻量测试通过不表示模型性能已经改善。
