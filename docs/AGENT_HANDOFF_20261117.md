# 03_PRECODER 当前进度与接替清单（2026-11-17）

本文供新的 agent 独立接替，不要求阅读原聊天。事实分为：GitHub 实时核查、用户回传的服务器输出、尚未验证的推测。本文不包含代理凭据或 GitHub token。

## 1. 先回答用户最关心的三个问题

**截至2026-11-17 09:44 UTC（柏林10:44 CET）的GitHub核查，新机CE模型、旧机本轮low-SNR报告、旧机本轮完整模型，均未在本仓库约定发布位置上传成功。源码同步已经完成。**

| 内容 | 确认状态 | 证据 |
|---|---|---|
| 新旧机 main 源码 | 已同步 | 用户两机HEAD和远端main均为 `eee870a1cf3cd229c919c54c8ee99da21ff6ad1b` |
| 旧机清理记录与.gitignore | 已上传 | 上述提交，7文件，真实Git push成功，包约31.67KiB |
| 旧机本轮low-SNR完整报告 | 未上传成功 | 约16.64MiB Git push反复403；结果分支不存在，main也没有该报告 |
| 新机本轮CE完整报告 | 未上传成功 | 约3.28MiB Git push返回403；结果分支不存在，main也没有该报告 |
| 旧机本轮完整模型 | 未上传成功 | 模型Release尝试返回 `network_or_tls_failure`；仓库当前无Release |
| 新机CE65000两个完整模型 | 未上传成功 | 仓库当前无模型Release，main无checkpoint；未收到新机模型上传成功记录 |
| 旧机本地模型归档 | 有prepare/verify已通过的流程证据，但未独立审计清单 | 用户已进入独立Release publisher并在网络阶段失败；不得写成已上传 |
| 新机本地模型归档 | 未确认 | 未取得prepare完成或文件清单输出，不能假定存在 |
| 最新旧机模型是否已迁入新机 | 未确认 | 没有本次归档rsync/restore成功输出 |

核查方法：认证GitHub API分页列举分支与Releases；递归读取main树（`truncated=false`）。当时没有任何 `results/` 分支，Release列表为空，main树无 `runs/`、模型权重或本轮报告。历史结果曾经上传并经审阅删除分支，与本次缺少归档不是同一件事；不要说历史从未上传。

可复核的远端快照见 [AGENT_HANDOFF_GITHUB_AUDIT_20261117.json](AGENT_HANDOFF_GITHUB_AUDIT_20261117.json)，其中单独记录核查时间、main提交、两条预期结果ref、Release及草稿数和权重路径。它不包含SSH本地文件或认证信息。

本轮预期结果分支：

```
results/twc_v1_k8_low_snr_interference_v1/63d630146f51
results/twc_v1_k8_ce_refine_v1/9e9d46c5e8f7
```

## 2. 用户目标与操作边界

- 只允许修改 `Charlie-Su-55/03_PRECODER`。08参考项目、论文、Overleaf、人工绘图、历史结果保留。
- 新机作为主要训练和综合评估机器，旧机保留历史并承担辅助独立实验。
- 用户要求报告直接通过GitHub读取，不再手动将ZIP上传聊天。标量报告与完整模型是两类交付。
- 云端只读源码、编辑代码/文档、运行小规模CPU检查；不连接SSH、不启动正式训练、不运行CUDA/Sionna、不加载生产checkpoint。服务器操作由用户执行。
- 不追加FDMA训练或优化。已有FDMA权重冻结；如为机制控制做前向重评，应明确新增更新为0。
- 用户不接受反复操作无新证据的网络排查。不要再原样重试已连续403的命令，把重复当作修复。
- 保持原研究协议、固定末态选择和test封存。完整CSV审计未完成，不能把终端摘要写成完整结果核验。
- 不自动删除历史run、覆盖本地文稿、裁剪模型来源链、修改receipt，或放宽checkpoint身份校验。

## 3. 两台机器与文件位置

两台repo物理路径刻意一致：`/home/r50037942/project/03_PRECODER`。

| 项目 | 旧机 | 新机 |
|---|---|---|
| 登录用户 / 主机标识 | `r50037942` / `w00611266-ELITE` | `rong` / `admin1-ELITE` |
| 用户HOME | `/home/r50037942` | `/home/rong` |
| Conda环境 | `precoder` | `precoder` |
| Python | `/home/r50037942/.conda/envs/precoder/bin/python` | `/home/rong/miniforge3/envs/precoder/bin/python` |
| 当前本批训练 | 已完成low-SNR两轨 | 已完成CE-refine两轨 |
| 预定模型归档目录 | `$HOME/precoder_model_releases/old_20261117` | `$HOME/precoder_model_releases/new_20261117` |

新机曾通过用户执行的真实CUDA/Sionna/RZF环境检查，以及早期四批实验的历史证据和六个完整模型严格加载验收。环境包括Python3.12.13、torch2.11.0+cu130、numpy2.4.4、Sionna2.0.1、RTX5090。**这不代表最新joint/direction/low-SNR模型已经迁移。**

最后一次两机终态输出为2026-11-17约09:08 UTC：两批均 `completed | stage=directory_export`，对应训练进程不存在、tmux会话已结束。没有需要因上传失败重启的训练。此后没有本agent批准或启动新的科学训练。

## 4. 旧机：low-SNR干扰A/B已完成

- Namespace / 根目录：`twc_v1_k8_low_snr_interference_v1` / `runs/twc_v1_k8_low_snr_interference_v1/`。
- 实际训练源码提交：`31c1fb30e6e2995d847a02226a6cbf19ff86b5b0`。
- 两条轨迹：`low_snr_control`、`low_snr_interference`，从相同Hybrid direction末态**独立分叉**，各新增10000 UE+BS有效更新，不串联。
- micro4×累积8=32完整MU；每更新75%低gamma0/5/10，25%高gamma15/20/25。
- 两轨均完成10000；各EMA从40000到50000。UE、BS有梯度；CE、误差表、FDMA冻结。
- 对照损失：`-complete-MU sum-rate + 100*phase/norm-invariant direction error`。
- 实验额外加入低gamma归一化**下行**泄漏项。不是上行源分离损失。
- 当前UE没有SNR输入；训练采样随gamma变化，不等于UE发送自适应。
- 最终仍等权验证六个gamma节点、固定末态、单源训练seed、test封存。

关键目录：

```
runs/twc_v1_k8_low_snr_interference_v1/training/low_snr_control/
runs/twc_v1_k8_low_snr_interference_v1/training/low_snr_interference/
runs/twc_v1_k8_low_snr_interference_v1/reports/attempt_63d630146f51/
runs/twc_v1_k8_low_snr_interference_v1/uploads/63d630146f51.json
```

终端回传的和速率（bps/Hz）：

| gamma/dB | 冻结FDMA | Hybrid起点 | low-SNR对照 | 加干扰项 |
|---:|---:|---:|---:|---:|
| 0 | 29.157496 | 26.730589 | 27.019171 | 27.027833 |
| 5 | 30.410539 | 29.661101 | 29.836368 | 29.843585 |
| 10 | 30.819421 | 30.772262 | 30.903108 | 30.905639 |
| 15 | 30.955967 | 31.087470 | 31.200970 | 31.201843 |
| 20 | 30.998133 | 31.181200 | 31.283282 | 31.284164 |
| 25 | 31.009143 | 31.204351 | 31.304128 | 31.303882 |

- 对照减起点：六点平均+0.151676，配对batchCI=[+0.143063,+0.160288]。
- 干扰项减对照：低三点+0.006137，CI=[+0.004567,+0.007707]；六点+0.003320，CI=[+0.002100,+0.004540]；高三点+0.000503，CI跨零。
- 干扰项减FDMA：低三点−0.870133，高三点+0.275549，六点−0.297292。
- 判断：低SNR续训有收益；当前泄漏项额外作用很小。10–25dB均值高于冻结FDMA，主要差距在0、5dB。10dB逐点CI尚未取得，不能直接声称逐点显著。
- Hybrid累计训练更多；不能把与冻结FDMA的比较写成等训练预算。全部负结果保留。

尚未拿到并独立重算本批完整samples/summary/comparisons CSV、逐点CI与最新机制分解；以上是终端证据及对其数值的计算。

## 5. 新机：CE-refine已完成

- Namespace：`twc_v1_k8_ce_refine_v1`。
- 实际训练源码提交：`63b7467ad452523b2a6efc98e89828f430fbe26d`。
- `nmse`与`geometry`各从原CE5000独立新增60000更新，末态均CE65000，不是串联成125000。
- NMSE对照 vs NMSE+方向+用户间归一化复Gram，两个辅助系数固定1。
- AdamW LR1e-4 cosine至1e-5，micro4×累积8，六节点均衡。UE、BS codec、FDMA、历史EMA新增更新均0。
- 每5000保留不可变checkpoint与monitor，包含0点，共13套/轨；最终固定末态。
- 三个CE模型各自独立校准误差表，先锁定再最终验证；最终六节点与五个中点。
- 原CE5000预训练已经连续覆盖0–25dB；中点只表示本次六节点续训的off-node插值，不是全历史未见SNR。

关键目录：

```
runs/twc_v1_k8_ce_refine_v1/training/nmse/
runs/twc_v1_k8_ce_refine_v1/training/geometry/
runs/twc_v1_k8_ce_refine_v1/reports/attempt_9e9d46c5e8f7/
runs/twc_v1_k8_ce_refine_v1/uploads/9e9d46c5e8f7.json
```

终端回传的是**六节点等权均值，不包含五个中点**：

| 模型 | UL NMSE线性 | 方向误差 | 用户间Gram误差 |
|---|---:|---:|---:|
| source_ce5000 | 0.04719672 | 0.04124292 | 0.003653126 |
| nmse65000 | 0.03180070 | 0.02913728 | 0.001987503 |
| geometry65000 | 0.03706585 | 0.03012284 | 0.001588453 |

NMSE续训分别降32.62%/29.35%/45.59%；geometry分别降21.47%/26.96%/56.52%。geometry比NMSE的Gram再低20.08%，但UL NMSE高16.56%，不能直接选geometry为全系统最好模型。

**新CE尚未替换进最新Hybrid做实际codec/downlink验证；不能写成已提升下行和速率。** 独立CE实验的shared解析恢复误差使用均匀符号prior，只是离线proxy，不是实际codec恢复或sum-rate。

本批完整CSV、逐SNR/中点曲线、校准和收敛曲线尚未独立审计。

## 6. 原协议、模型与训练来路

固定K8、BS32天线、256子载波；总反馈256RE=64pilot+192data；每UE能量64=pilot16+data48。FDMA每UE24独占复坐标；HybridG4为4个2UE组，每UE4独占key+40组内共享坐标。`N0=.25*10^(-gamma/10)`，RZF alpha=.03，下行25dB，六gamma0/5/10/15/20/25。

数据是Sionna CDL-C独立UE采样，UL/DL另行独立生成，不是COST2100，也未建立邻近UE共享随机散射体或FDD帧级互易边信息。不要凭共用统计协方差宣称跨UE信源相关。

UE：自己的完整DL CSI→angular-delay变换→有序Swin编码器→选码字联合能量归一化→模拟反馈。AD是angular-delay，不是A/D数字化。当前UE无gamma输入。

BS的UL CE：pilot-only rank8 Fourier-LS初值+残差U-Net，已有公共gamma输入。正常接收：estimated-UL与CE方差/历史power-EMA驱动的正则化/LMMSE风格soft恢复→两层Joint+spatial cross-attention+四层Swin→CSI形状输出→固定RZF得到W。当前主干不是纯网络直接输出W。soft恢复后的残差在latent层，不能描述成“先LMMSE估CSI再做CSI残差”。

rate-only阶段的CSI形状输出不必是物理CSI的最好估计；raw NMSE改善不自动意味着干扰投影或sum-rate改善。

主来路，勿把并行对照累加进主模型：

1. CE5000独立预训练，随后原CE冻结。
2. 每臂UE+BS codec20000，CSI重建目标。
3. 每臂BS-only rate5000。
4. 每臂UE+BS joint10000；FDMA在此末态冻结。
5. Hybrid direction额外10000。
6. 当前两条low-SNR轨迹各再10000，由同一方向末态独立分叉。

每条当前Hybrid终态累计UE50000、BS55000、EMA50000；冻结FDMA累计UE30000、BS35000、EMA30000。新CE65000仍是独立候选，未自动替换旧链。

## 7. 既有科学发现，避免重复无效路线

- 早期ICC-aligned Hybrid共享恢复的CE误差经接收滤波放大；仅按总CE方差报告可靠度会低估实际soft误差。
- 先前true-UL实验是特权敏感性诊断，不是可部署臂，也不保证是性能上界。
- 固定滤波器下的真实UL条件误差传播能较好解释实际soft误差；真UL只用于离线评分。
- 校准/soft NMSE改善在冻结decoder时曾导致速率下降，说明接口/任务目标重要；不能只按NMSE选接收机。
- UE+BS joint rate训练两臂都改善，但FDMA曾改善更多；“增加端到端训练”本身没有解决Hybrid低SNR差距。
- W残差相对同预算RZF续训只有约0.02–0.03bps/Hz增益；这不是纯直接W实验，不能推断所有直接W方案无效。
- direction辅助及当前low-SNR续训有小幅收益，但显式DL泄漏额外收益很小。主要待查仍是低SNR的估计/叠加恢复与下游干扰管理。
- 自适应共享门控、24坐标正交基础层+额外共享增强层等只讨论过，**尚未实现、未训练、未验证**。没有新的论文contribution已被证明的结论。

## 8. 已实现的上传/迁移工具与已知问题

模型交接PR[#20](https://github.com/Charlie-Su-55/03_PRECODER/pull/20)已合并；工具提交 `6bdf55fe4fc015d8e0787ca267195f6c1eed75a0`，合并提交 `8e45a2c3c7aa55ea632799ed0f4fb9107d4a198a`。随后用户提交main的.gitignore和清理记录，得到eee870a。

| 工具 | 实际作用 |
|---|---|
| `publish_twc_v1_k8_low_snr.py` | 校验并将标量报告推到独立结果分支，不含权重 |
| `publish_twc_v1_k8_ce_refine.py` | 同上，对应CE报告 |
| `tools/model_release_03.py` | stdlib归档/校验/恢复，不解pickle、不导入torch、不启动GPU |
| `tools/publish_03_model_release.py` | 流式上传归档分卷和manifest到Release，服务器SHA256全通过后发布 |
| `tools/publish_completed_models_20261117.sh` | 将报告push、归档、Release上传顺序串联；**首步失败就退出** |

归档是未压缩tar的最多1GiB分卷，不是ZIP。old配置选择固定12类实际存在且完成的科学run，含历史模型与最新Hybrid来源链；new配置包含完整CE-refine run及原CE训练源目录。清单每文件、每分卷SHA256；恢复时已有同SHA文件跳过，有冲突预检即停止，新增文件原子写入；不覆盖、不删原结果。metadata中的原训练提交和来源路径原样保留。

工具真实CPU检查：26项全部通过，覆盖实际tar分卷往返、损坏、冲突与路径保护、流式HTTP请求体、草稿重试、服务器digest失败、TLS/凭据处理。Release事务使用API fixture；**没有在云端做真实大资产上传或生产模型加载**。CPU通过不证明企业网关允许上传。

已知工具不足：

- shell一键脚本把报告上传放在归档之前，导致403时后续本地归档也没执行。独立归档不应该被公网报告push阻断。
- Release工具将连接/超时/证书等统一为 `network_or_tls_failure` 并隐藏原错误；目前诊断粒度不足，根因尚未确定。
- 大权重不能直接 `git add runs`：仅codec的float32参数已约123MiB，完整checkpoint含optimizer更大。不能把当前3–17MiB报告push403简单解释为Git的100MiB单文件限制。

## 9. 上传失败证据与禁止重复的错误判断

已观察到：

1. 旧机low-SNR报告Git push约16.64MiB失败HTTP403；新机CE报告约3.28MiB也失败403。
2. 用户改为只有旧机单独上传同一low-SNR报告，仍然403。因此“仅两机并发造成全部失败”没有被验证，串行并未解决这一次问题。
3. 旧机31.67KiB的main提交真实push成功，说明那次认证与Git小请求链路可用；不证明更大的报告POST或Release上传可用。
4. 旧机独立模型Release publisher报：`category=network_or_tls_failure,status=null,response_kind=null,github_request_id_present=false`；无成功Release URL或资产校验记录。
5. 用户用curl显式指定旧机loopback3128与系统CA：api.github.com根路径返回200及GitHub request ID；uploads.github.com根路径返回302至github.com及GitHub request ID。说明当时这些HEAD请求可达且TLS通过，未测试token写权限或大文件POST。
6. Conda Python默认证书与curl显式系统CA不同，是曾提出的猜测，尚未证实。临时强制系统CA/两域走3128的建议没有回传成功输出。

前agent让用户重复了同一失败报告publisher，未获得新诊断；应停止此循环。不要把 `Writing objects:100%` 或失败后的 `Everything up-to-date` 当上传成功，也不要把GET、dry-run或源码同步成功当报告/权重已上传。

新机曾因把代理域密码误输成sudo密码造成问题，用户后来修正且Git读取正常。这是历史事实，**不能不经验证就归为本次原因**。不改永久代理配置、关闭TLS或复制凭据来碰运气；不索取用户token/密码或原始认证trace。

## 10. 本地未跟踪材料与迁移保护

用户最后一次两机检查：本地HEAD与远端main一致，两个git diff均空。旧机仍有：

```
docs/TWC03_SSH_HANDOFF_20261102.md
docs/TWC_1_0_K8_two_week_plan_20261103_V2.md
docs/figures/
overleaf/
paper/data/as_spatial_verified/
paper/data/k4_swin_adaptation_verified/
```

新机显示 `?? overleaf/`。这些是未跟踪材料，不是代码冲突，也不能说明两台同名目录字节相同。不得 `git add .` 连同文稿或私有配置自动公开。新机刚才用字面占位符 `git add <要提交的文件或目录>` 导致语法错误，又因未配作者身份commit失败；随后pull成功，并没有新增新机提交。

最新模型迁移必须保留整个来源链：

- 两条low-SNR各含latest checkpoint、manifest、terminal receipt、initialization、progress，以及 `source_identity.frozen_files` 递归列出的原CE、codec、校准器等。
- CE严格加载会检查每轨0..60000每5000的13个 `.pt`及`.json`，以及13个monitor目录中的manifest/summary/input fingerprints；只复制latest四文件不够。
- CE候选独立的calibration/报告记录也须保留；不用真UL数据作部署输入。
- `git pull`只同步代码，不下载Release资产，也不会把旧机ignored runs搬到新机。
- GitHub归档阻塞不妨碍一次内网文件迁移；具体内网地址让用户本地确认，不在公开文档记录认证或私有网络配置。
- 迁移后仍需用户在新机用现有严格加载器验收真实模型。tar字节校验不能替代该验收。

## 11. 接替者的建议优先顺序

- [x] 两批训练均完成；源码两机与main同步。
- [x] 已回传终态摘要；已做上述初步分析。
- [x] 模型归档/Release工具已合并，CPU检查通过。
- [ ] 先保护并核验两机本地完整模型、收据、报告、已有归档；不要重训或删除。
- [ ] 分别解决本轮标量报告读取与模型归档，不让公网失败阻塞新机接管。
- [ ] 核验最新旧机模型依赖链迁入新机，并实际严格加载；保留旧机原件。
- [ ] 审计完整CSV：来源/哈希、配对输入、计数、六节点、中点、学习曲线与区间，保留全部负结果。
- [ ] 在新机设计**零更新**的旧CE/NMSE65000/geometry65000配对替换，各配自己独立已锁定误差表；固定UE波形、信道、pilot、噪声、历史prior、RZF，检查实际soft恢复、CSI方向、下行sum-rate。
- [ ] 再决定是否对Hybrid做接收端适配或修改发送表示。新CE改变decoder输入/置信度分布，冻结速率可能下降，不能只凭CE NMSE下结论。
- [ ] 后续新增训练需用户明确预算；不追加FDMA训练，不把讨论中的新架构写成已实现。

## 12. 阅读入口与给新agent的起始要求

建议先读本文件，再按需读：

- [ICC-aligned协议与启动](TWC_V1_K8_ICC_ALIGNED.md)、[参考对应](TWC_V1_K8_ICC_ALIGNED_REFERENCE.md)、[历史状态](TWC_V1_K8_ICC_ALIGNED_STATUS.md)
- [UE+BS联合训练](TWC_V1_K8_UE_BS_RATE.md)、[方向辅助](TWC_V1_K8_HYBRID_DIRECTION.md)
- [本批low-SNR](TWC_V1_K8_LOW_SNR.md)、[本批CE-refine](TWC_V1_K8_CE_REFINE.md)
- [模型归档与迁移说明](MODEL_RELEASE_HANDOFF_20261117.md)、[代码布局](CODE_LAYOUT.md)

用户可直接给接替agent：

> 先阅读 docs/AGENT_HANDOFF_20261117.md。两批训练已完成，代码已同步；本轮报告与模型GitHub归档尚未成功。请先核查真实已存在的本地/远端成果并保护它们，再处理报告读取和新机接管；不要重复失败命令、重新训练、优化FDMA或修改论文。云端不连接SSH、不运行正式CUDA训练，服务器命令由我执行。

本次交接提交只增加文档，不修改科学源码、不改变训练状态、不尝试重新上传服务器模型或报告。
