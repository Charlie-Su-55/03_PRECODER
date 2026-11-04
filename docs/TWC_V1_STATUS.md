# TWC 1.0 K8 科研进度

最后核查：2026-11-04 15:40 CET（本机时间）
依据：本地仓库源码、Git状态、配置/输出目录、进程和GPU快照，以及 Overleaf 快照与编译产物。本文不把旧协议结果当作新 K8 证据。

## 一屏摘要

- **当前阶段：** 按V2完成K8数据隔离/重放入口、LS初值screen和LS→U-Net CPU链路检查；首轮CE输入契约已固定。Joint接收、完整训练器、预编码和正式数据隔离审计仍未完成。
- **状态：** 数据/LS/U-Net接口 **完成并通过 CPU 检查**；K8训练 **未开始**，本轮优化更新0。
- **GPU快照：** 本轮未查询或使用GPU；最近历史快照仍为11:51 CET（0%、18 MiB / 32607 MiB），不可代表当前状态。未来每次GPU工作前重新检查；占用超过20%先请示。
- **代码版本：** 本轮K8数据/LS/CE CPU接口已推送至`e5a27e5`。远程为`https://github.com/Charlie-Su-55/03_PRECODER.git`。
- **未提交变更：** K8工作已提交推送；原有`.gitignore`修改、5个旧源码删除、清理回执、SSH交接文档、Overleaf副本、历史数据及本地V2计划文件仍留在工作区，未纳入本轮。
- **代码/验收/证据区分：** K8分角色CDL回放、pilot-LS＋插值、Ep条件化pilot-only U-Net及G=1/2/4 Swin接口**代码已实现**；小批CPU回放和接口验收**通过**；CE训练/校准、Joint接收器/RZF及独立性能证据**未开始/没有**。
- **训练步数/checkpoint：** K8本轮更新数 0 / 暂定计划预算200k codec更新、约5k CE更新（尚未进入训练配置验收）；本轮loss/LR未运行；来源累计步数：不适用/未知；最后保存且核验的K8 checkpoint：无。没有仅凭日志声称权重已核验。
- **训练曲线和性能：** K8固定验证集DL NMSE、真实下行和速率、loss阶段/权重/学习率均**未知（尚未训练）**；六SNR×两个seed正式测试、配对差值与置信区间均**未知**。
- **资源与隔离验收：** CPU smoke核对G=1/2/4映射总预算偏差0 RE；所用候选每UE数据能量最大误差`7.63e-6`。gamma=10 dB时N0=0.025；经验噪声方差、其他gamma链路和train/validation/test隔离仍未知。正式点是反馈gamma 0/5/10/15/20/25 dB，DL SNR固定25 dB。
- **论文：** 发现主稿`TWC_progress_updated_analog_superposition.tex`。最新构建PDF为`overleaf/build/TWC_progress_updated_analog_superposition.pdf`，2026-11-04 11:23:50 CET生成，18页；构建成功但有warning，详见下文。该状态盘点没有改tex。
- **主要阻塞：** 真实K8 CDL的LS→未训练U-Net前向已在CPU跑通，但CE训练/校准、反馈软恢复接入Swin、Joint CSI重建/RZF和完整数据集隔离审计仍缺，因此尚无K8 learned-link性能证据。

## 阶段状态

统一状态仅使用：**未开始、运行中、完成、失败、跳过**。代码、运行验收、性能证据分别标注，避免把实现等同于实验结论。

| 工作项 | 状态 | 代码实现 | 运行验收 | 独立性能证据 | 当前依据/缺口 |
|---|---|---|---|---|---|
| K8场景、RE预算与SNR点对齐 | 完成 | 文档基线完成 | 算术核对完成；程序映射未开始 | 不适用 | 见`TWC_V1_PROTOCOL_BASELINE.md`；Hybrid预算每种G均256 RE，FDMA为64+8×24=256 RE |
| K8布局/预算配置模块 | 完成 | 完成（协议层） | CPU检查完成 | 不适用 | `utils/twc_v1_k8_protocol.py`与`configs/twc_v1_k8_protocol_config.py`；不是训练就绪实现 |
| Ep/Ekey/EAS候选网格 | 完成 | 六个正值候选已配置 | CPU约束检查完成 | 未开始 | Ep={8,16,24}；key:shared={1:3,1:1}；等验证集选优 |
| K8 CPU物理反馈链路 | 完成 | pilot、LS插值、数据发射、key/shared MMSE软恢复已实现 | `preflight_twc_v1_k8_link.py`通过，G=1/2/4 | 未开始 | 插值不是U-Net；不含CSI重建、预编码或性能结论 |
| K8 pilot-only U-Net接口 | 完成 | 08共享频率残差U-Net适配为K8/64 pilot RE、03噪声口径；输入含N0/gamma与显式Ep | 随机输入边界及真实CDL LS→U-Net共18种gamma×Ep接口前向通过 | 未开始 | U-Net未训练；LS初值由调用方仅基于pilot构造 |
| K8 Swin UE编码/BS解码接口 | 完成 | 复用03内Swin窗口块；共享UE权重、显式G条件、key4+shared 160/80/40 | CPU前向G=1/2/4输出有限且维度正确 | 未开始 | 尚未接入真实反馈软恢复、Joint模块或RZF；新增结构无训练证据 |
| K8 CDL分角色数据与重放入口 | 完成 | 新增CE训练/codec训练/CE校准/验证/测试独立seed role和样本地址指纹；复用Sionna RNG隔离 | Sionna 2.0.1 CPU生成五角色各1对K8 CDL；固定验证与训练step地址可复现；未发现角色间指纹碰撞 | 不适用 | 仅小样本地址/重放验收；完整训练checkpoint暂停恢复与大规模数据审计未知 |
| LS初值方法screen | 完成 | 新增归一化Gram ridge Fourier-delay LS(rank4/rank8)，保留pilot-LS线性插值 | 32帧、6 gamma×3 Ep固定validation screen；线性插值18/18条件胜出 | 有限validation设计证据 | 小样本只用于初值设计，不能当正式UL性能结果；首轮CE暂定固定线性插值 |
| 完整K8协议/候选冻结 | 运行中 | 几何、六点能量候选已配置 | CPU链路烟雾检查通过；gamma查表策略未选 | 未开始 | 数据namespace、pilot gamma查表和UL状态选优仍待验证集核定 |
| CPU配置、预算和数据隔离检查 | 运行中 | 资源/能量约束已配置 | 单次CPU smoke通过；数据隔离未知 | 不适用 | 当前为合成CPU输入，真实可回放数据接口尚未接入 |
| Stage A：LS与LS＋U-Net UL CE预训练及校准 | 运行中 | LS方法screen暂定冻结为pilot-LS＋线性插值；U-Net输入含pilot mask、LS初值、N0/gamma、Ep；监督目标应为H_UL | 真实CDL端到端前向CPU验收通过；trainer/校准未实现 | 未开始 | 训练约5k更新未运行；当前screen不等于U-Net收益证据 |
| 完整端到端CSI反馈到下行预编码 | 未开始 | K8 Swin/CSI重建/RZF链路未实现 | 未开始 | 未开始 | 当前smoke仅验证UL导频和key/shared物理反馈接收 |
| Stage B：FDMA/Hybrid NMSE warmup | 未开始 | 未知 | 未开始 | 未开始 | 计划每模型/seed 10k；目标K8 checkpoint、日志与曲线未找到 |
| Stage C：匹配NMSE/联合loss终段 | 未开始 | 未知 | 未开始 | 未开始 | 计划每分支20k；权重日程、loss曲线与匹配比较未开始 |
| Stage D：资源/能量策略、RZF及WMMSE诊断 | 未开始 | 未知 | 未开始 | 未开始 | 计划先验证集查表/扫描；新协议策略与结果未找到 |
| 六SNR、两个seed独立正式评估 | 未开始 | 未知 | 未开始 | 未开始 | 独立测试、配对样本和置信区间材料未找到 |
| Overleaf主稿构建链 | 完成 | 不适用 | 完成（构建成功） | 不适用 | 有最新18页PDF；页面人工检查未知。主稿尚未按K8计划改写 |

## 运行、更新数与checkpoint

- 当前运行状态：训练**跳过**（本次没有训练进程；不是训练失败）；K8 CDL/CE接口CPU preflight**完成**。本轮未读取GPU快照。
- K8新实验本轮更新数：**0**。训练尚未开始；计划预算暂按CE约5k、codec 4份warmup×10k加8个终段×20k（共200k codec更新）记录，尚未配置化验收。
- 来源checkpoint累计步数：**未知/未适用**。新K8训练未声明或核验任何初始化来源。
- 最后保存并核验的K8权重步数：**无**。现有旧K4权重/结果不属于K8协议，且本次未加载核验。
- 训练日志步数与权重步数：目前没有K8训练日志；无可报告日志步数。
- 首个500–1000步耗时、剩余时间估算、训练峰值显存：**未知**。待GPU验收阶段按真实吞吐计算；之后不要求每500步停下确认。

## 配置与协议验收

| 项目 | K8计划值/规则 | 当前实际验收 |
|---|---|---|
| 用户数、天线、子载波 | K=8、Nt=32、Nsc=256 | 未开始；仓库中`grouped_codec_train_config.py`和`grouped_hybrid_power_train_config.py`的实验配置为K=4。旧`exp_8ue_128sc.py`是不同的128子载波旧方案，不能作本轮验收 |
| 反馈预算 | 256 RE；64 pilot RE；每UE key d=4 | CPU链路映射通过；总占用偏差0 RE |
| 分组预算 | G=1/2/4；每组共享维数a=160/80/40；预算64+8d+Ga=256 | CPU链路 smoke通过；RE偏差0；能量映射最大误差7.63e-6 |
| FDMA比较 | 不含key；64 pilot RE + 每UE24数据坐标×8 | 几何/预算映射CPU检查通过，偏差0 RE；FDMA无线链路尚未在smoke中执行 |
| SNR点 | 正式反馈gamma=0/5/10/15/20/25 dB；DL SNR固定25 dB | 配置已记录；CPU smoke只运行gamma=10 dB，N0=0.025 |
| DL/UL与噪声 | UE完美DL CSI；BS使用LS＋U-Net UL CSI；N0=0.25×10^(-gamma/10) | CPU真实CDL在gamma=0/5/10/15/20/25与Ep=8/16/24上完成LS→未训练U-Net有限值前向；复噪声经验方差统计尚未做 |
| 能量 | 每UE总能量64；候选Ep={8,16,24}，key:shared={1:3,1:1} | 六个候选合计64；CPU smoke单个候选实际数据能量最大绝对误差7.63e-6；尚未验证集选优 |
| 数据隔离 | 独立CE训练、codec训练、CE校准、验证、测试namespace；测试仅用于最终报告 | 五个角色CPU样本地址各异，固定validation和ce_train step地址可重复；5对中跨角色SHA碰撞0。完整namespace规模审计、正式checkpoint恢复检查未知 |
| 控制时序 | pilot后先估UL，再选择组员/数据能量并通知UE，随后发key和共享数据 | CPU smoke按pilot调用、LS估计、数据调用顺序执行；无控制消息开销模型；相干假设为UL在两阶段间不变 |

## 结果和材料状态

以下“未知”表示本次没有找到并核验满足新K8协议的材料，不代表结果为零或通过。

| 材料 | 状态 | 关键数值/偏差 | 所需证据 |
|---|---|---|---|
| LS、LS＋U-Net UL NMSE–SNR曲线 | 运行中 | 有32帧K8 CDL validation设计screen：线性插值平均per-link NMSE ratio 0.05975；Fourier-delay rank4 0.14266、rank8 0.18292（18条件等权平均）。U-Net未训练，无其NMSE | `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_LS_validation_screen_20261104.csv`；后续需正式CE训练/独立校准曲线 |
| 固定验证集DL NMSE和真实下行和速率曲线 | 未开始 | 未知；loss阶段、lambda权重和LR未运行 | 每步/验证日志、配置、权重和阶段标记 |
| 正式评估六SNR×两seed | 未开始 | 六点均未知 | 独立测试CSV及run ID |
| 相对匹配FDMA、NMSE控制的配对差值/置信区间 | 未开始 | 未知 | 同一测试信道实现上的逐样本结果及统计脚本 |
| 信号/干扰功率、能量分配、分组选择统计 | 未开始 | 未知 | 评估样本级CSV及选择器配置 |
| 论文数值与数据追溯 | 部分（旧稿材料存在） | 新K8数字未知 | 新K8 CSV、run ID；旧稿图表不可自动视作新证据 |

仓库中存在多份K4/K16旧协议CSV和论文图表；本状态不将它们并入K8对照。旧结果保留以便追溯，最终需按协议和run ID区分。

## Overleaf实际状态

- 主稿：`/home/r50037942/project/03_PRECODER/overleaf/TWC_progress_updated_analog_superposition.tex`，IEEEtran journal，pdfLaTeX；目录中还存在模板和旧稿，按上传清单/当前构建配置确认此文件为当前主稿。
- 最新生成PDF：`/home/r50037942/project/03_PRECODER/overleaf/build/TWC_progress_updated_analog_superposition.pdf`；pdfinfo显示18页，生成时间2026-11-04 11:23:50 CET，786551字节。
- 编译日志：`/home/r50037942/project/03_PRECODER/overleaf/build/TWC_progress_updated_analog_superposition.log`；LaTeX Workshop已设onFileChange、3秒间隔、build输出目录、tab预览器。
- 编译结果：日志包含`Output written on ... (18 pages)`，构建成功。主要warning：未识别全局选项`lettersize`；pgfplots处于backward compatibility模式；2处Overfull hbox（约9.02pt、2.55pt）；多处Underfull hbox/vbox；amsmath提示公式起始处方括号组。未发现未定义引用/文献警告；视觉检查状态未知。
- 保留的上传基线PDF：`/home/r50037942/project/03_PRECODER/overleaf/TWC_progress_updated_analog_superposition.pdf`，18页，原始上传快照manifest记录SHA256；与build下新生成文件分开保留。
- TWC K8计划论文数字：未开始；目前无可对应的新K8 CSV/run ID。旧稿里数字需逐项追溯，不能作为新协议性能证据。

## 源码、Git与证据路径

- 仓库：`/home/r50037942/project/03_PRECODER`；GitHub origin：`https://github.com/Charlie-Su-55/03_PRECODER.git`。
- 最近已推送commit：`e5a27e5`（K8 replayable CDL、LS初值screen与CE CPU preflight）。
- 当前未提交源码变更：无本轮K8源码待提交；工作区仍有`.gitignore`修改及5个旧文件删除`diagnose_cross_cdl_condition.py`、`old/diagnostic_eval.py`、`old/train_adaptive_8ue.py`、`old/train_csinet_baseline_8ue.py`、`old/train_swin_baseline_8ue.py`。这些既有变更及其他未跟踪文件未纳入本轮备份提交。
- **2026-11-04 14:31 CET：** 选择性提交并推送K8候选能量网格、CPU反馈链路和preflight；commit `629ae85`，远端`main`从`72f6792`前进。`git diff --check`通过；原有旧文件删除、`.gitignore`和未跟踪目录未纳入。
- `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_LS_validation_screen_20261104.csv`及同名JSON记录32帧screen及数据指纹；该结果不是正式CE曲线。

最多4个关键证据文件：

1. `/home/r50037942/project/03_PRECODER/docs/TWC_1_0_K8_two_week_plan_20261103_V2.md`
2. `/home/r50037942/project/03_PRECODER/utils/twc_v1_k8_replayable_data.py`
3. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_LS_validation_screen_20261104.csv`
4. `/home/r50037942/project/03_PRECODER/preflight_twc_v1_k8_ce_chain.py`

## 问题与下一步

1. **K8训练链路仍未打通。**新增U-Net/Swin目前只通过CPU前向，不含CE trainer/calibrator、软恢复联合解码或RZF。影响：尚不能训练模型或生成K8重建/速率结果。
2. **08接口不能原样移植。**08 U-Net固定K4/32 pilot RE、噪声参考0.125；03需要K8/64 pilot RE及0.25参考。已适配维数和噪声换算，但没有迁移08训练/校准材料，也未证明跨设置泛化。
3. **选择策略和数据审计未完成。**六个能量候选已配置、样本角色namespace已实现；pilot gamma查表、UL条件下Ekey/EAS/G选择及大规模重叠/暂停恢复审计仍未完成。
4. **Git工作区已有旧源码删除和其他未跟踪内容。**后续备份只选择K8文件，不能混入原有删除、Overleaf或历史数据。
5. **主稿虽可编译，警告尚未处理，视觉预览未验收。**这不阻挡当前CPU协议工作；本轮按要求未编辑论文。

下一项动作：按V2补CE约5k更新训练器与独立校准入口；并行实现软恢复→Joint decoder→空间Swin→RZF及匹配FDMA，完成合并CPU验收。之后在GPU使用前检查实时占用，超过20%先请示；CE训练/校准需要GPU。08保持只读。

## 更新记录

- **2026-11-04 11:51 CET：**首次建立状态页。直接盘点Git、配置源码、实验输出目录、训练进程、GPU和Overleaf构建产物；未启动训练。记录了K8计划与现有K4代码/历史证据的差别，以及当前未提交改动。
- **2026-11-04 12:05 CET：**完成首轮场景、Hybrid/FDMA预算算术和反馈SNR点对齐；新建`docs/TWC_V1_PROTOCOL_BASELINE.md`。实测/代码映射、能量候选、噪声、数据隔离仍未验收；未使用GPU。
- **2026-11-04 12:09 CET：**新增独立K8预算/映射及配置模块；CPU核对静态编译、G分组数、Hybrid/FDMA总RE和六点N0公式均通过。未改旧K4训练模块、未启动GPU；K8模型训练集成仍未开始。
- **2026-11-04 12:09 CET：**K8协议源码及对应计划/状态文档提交为`b1573e2`，状态页更新提交为`016e489`；只纳入两份K8源码和三份K8计划/状态文档，未纳入原有删除或Overleaf目录。推送`origin main`成功，远端回执`ee58686..016e489`。
- **2026-11-04 12:09 CET：**状态记录修正并推送，远端从`016e489`前进至`bcb52ef`；未提交的旧删除、Overleaf、清理和历史数据仍保留在工作区。
- **2026-11-04 14:20 CET：**配置Ep/Ekey/EAS六候选，新增分散pilot的LS插值、key/shared发送及MMSE软恢复；`conda run -n precoder python preflight_twc_v1_k8_link.py`以CPU验证G=1/2/4均通过。该smoke未测性能，不含U-Net、DL CSI重建和预编码。新源码待选择性提交/推送。
- **2026-11-04 14:41 CET：**只读审阅08 U-Net/Swin源码与CPU审计材料（08当前commit `bfa0427`、工作区干净）；据接口在03新增K8 residual U-Net、G条件化Swin编解码器及CPU preflight。py_compile通过，CPU smoke通过：pilot=64 RE、data-only Y置NaN不影响U-Net输出、G=1/2/4码字和CSI形状正确。08材料记载102项CPU验收但0次优化更新；本轮没有使用GPU、训练、改08或产生性能证据。commit `b1fadcf`已推送到GitHub `main`。
- **2026-11-04 15:36 CET：**按V2完成CPU工作批：新增K8 Sionna2.0.1分角色可重放数据入口，五角色真实CDL CPU生成/重放、seed隔离及全局RNG恢复检查通过；新增rank4/rank8 Fourier-delay LS候选并用32帧validation×6 gamma×3 Ep比较，线性插值18/18条件最低（平均per-link NMSE ratio依次0.05975/0.14266/0.18292）。确定首轮CE初值暂用pilot-LS＋线性插值；U-Net加显式Ep输入；真实CDL LS→未训练U-Net在18条件CPU前向通过。发现并消除新preflight对`utils/__init__.py` GPU setup副作用的触发。训练更新0，未使用GPU，未做正式性能结论。CSV/JSON位于`docs/evidence/`。
- **2026-11-04 15:40 CET：**选择性提交并推送本轮11个K8数据/LS/CE源码、preflight、状态页及screen CSV/JSON，commit `e5a27e5`；远端`main`从`12a605a`前进。`git diff --cached --check`通过；未包含`.gitignore`、旧源码删除、V2计划文件、Overleaf或历史数据。
- 后续每个主要阶段完成或出现影响进度的问题时，追加带时间的状态变更、证据路径及下一动作；保留负结果和历史记录，不覆盖已报告事实。
