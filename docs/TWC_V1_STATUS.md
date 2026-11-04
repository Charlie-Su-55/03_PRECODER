# TWC 1.0 K8 科研进度

最后核查：2026-11-04 14:20 CET（本机时间）
依据：本地仓库源码、Git状态、配置/输出目录、进程和GPU快照，以及 Overleaf 快照与编译产物。本文不把旧协议结果当作新 K8 证据。

## 一屏摘要

- **当前阶段：** 第1–2天的CPU协议和基本链路搭建；K8预算/布局与六点能量候选已配置，pilot→插值LS→key/shared发射→组内软恢复的原型已在CPU跑通。U-Net、CSI重建、预编码和数据隔离未开始。
- **状态：** K8训练 **未开始**；CPU物理链路原型 **完成并通过 smoke**；没有启动训练。
- **GPU快照：** 最近一次快照为11:51 CET：GPU利用率0%、显存18 MiB / 32607 MiB。本轮新增工作只用CPU，未重新采集GPU快照。未来每次GPU工作前重新检查；占用超过20%先请示。
- **代码版本：** 最近已推送基线为`72f6792`；本轮新增K8链路修改尚未提交。远程为`https://github.com/Charlie-Su-55/03_PRECODER.git`。
- **未提交变更：** `.gitignore`已修改；5个旧诊断/旧训练源码文件处于删除状态；本轮K8协议、配置和链路源码处于未提交状态；另有清理回执、SSH交接文档、Overleaf副本及历史数据未跟踪。
- **代码/验收/证据区分：** K8预算/布局和六点能量候选**代码已实现**；CPU物理反馈链路 smoke **通过**；LS＋U-Net、UE Swin、BS CSI重建、预编码、正式训练/评估**未开始**；独立性能证据**没有**。
- **训练步数/checkpoint：** K8本轮更新数 0 / 计划预算未知（计划中的codec为4×10k warmup + 8×20k终段，共200k更新；CE另约5k）；来源累计步数：不适用/未知；最后保存且核验的K8 checkpoint：无。没有仅凭日志声称权重已核验。
- **训练曲线和性能：** K8固定验证集DL NMSE、真实下行和速率、loss阶段/权重/学习率均**未知（尚未训练）**；六SNR×两个seed正式测试、配对差值与置信区间均**未知**。
- **资源与隔离验收：** CPU smoke核对G=1/2/4映射总预算偏差0 RE；所用候选每UE数据能量最大误差`7.63e-6`。gamma=10 dB时N0=0.025；经验噪声方差、其他gamma链路和train/validation/test隔离仍未知。正式点是反馈gamma 0/5/10/15/20/25 dB，DL SNR固定25 dB。
- **论文：** 发现主稿`TWC_progress_updated_analog_superposition.tex`。最新构建PDF为`overleaf/build/TWC_progress_updated_analog_superposition.pdf`，2026-11-04 11:23:50 CET生成，18页；构建成功但有warning，详见下文。该状态盘点没有改tex。
- **主要阻塞：** K8的预算映射和CPU物理反馈原型已跑通；但真实K8信道/U-Net、CSI重建、预编码及数据隔离尚未实现，因此尚无K8性能证据。

## 阶段状态

统一状态仅使用：**未开始、运行中、完成、失败、跳过**。代码、运行验收、性能证据分别标注，避免把实现等同于实验结论。

| 工作项 | 状态 | 代码实现 | 运行验收 | 独立性能证据 | 当前依据/缺口 |
|---|---|---|---|---|---|
| K8场景、RE预算与SNR点对齐 | 完成 | 文档基线完成 | 算术核对完成；程序映射未开始 | 不适用 | 见`TWC_V1_PROTOCOL_BASELINE.md`；Hybrid预算每种G均256 RE，FDMA为64+8×24=256 RE |
| K8布局/预算配置模块 | 完成 | 完成（协议层） | CPU检查完成 | 不适用 | `utils/twc_v1_k8_protocol.py`与`configs/twc_v1_k8_protocol_config.py`；不是训练就绪实现 |
| Ep/Ekey/EAS候选网格 | 完成 | 六个正值候选已配置 | CPU约束检查完成 | 未开始 | Ep={8,16,24}；key:shared={1:3,1:1}；等验证集选优 |
| K8 CPU物理反馈链路 | 完成 | pilot、LS插值、数据发射、key/shared MMSE软恢复已实现 | `preflight_twc_v1_k8_link.py`通过，G=1/2/4 | 未开始 | 插值不是U-Net；不含CSI重建、预编码或性能结论 |
| 完整K8协议/候选冻结 | 运行中 | 几何、六点能量候选已配置 | CPU链路烟雾检查通过；gamma查表策略未选 | 未开始 | 数据namespace、pilot gamma查表和UL状态选优仍待验证集核定 |
| CPU配置、预算和数据隔离检查 | 运行中 | 资源/能量约束已配置 | 单次CPU smoke通过；数据隔离未知 | 不适用 | 当前为合成CPU输入，真实可回放数据接口尚未接入 |
| Stage A：LS与LS＋U-Net UL CE预训练及校准 | 未开始 | LS＋插值占位已实现；U-Net未实现 | 未开始 | 未开始 | 需要接入K8真信道、训练CE并独立校准 |
| 完整端到端CSI反馈到下行预编码 | 未开始 | K8 Swin/CSI重建/RZF链路未实现 | 未开始 | 未开始 | 当前smoke仅验证UL导频和key/shared物理反馈接收 |
| Stage B：FDMA/Hybrid NMSE warmup | 未开始 | 未知 | 未开始 | 未开始 | 计划每模型/seed 10k；目标K8 checkpoint、日志与曲线未找到 |
| Stage C：匹配NMSE/联合loss终段 | 未开始 | 未知 | 未开始 | 未开始 | 计划每分支20k；权重日程、loss曲线与匹配比较未开始 |
| Stage D：资源/能量策略、RZF及WMMSE诊断 | 未开始 | 未知 | 未开始 | 未开始 | 计划先验证集查表/扫描；新协议策略与结果未找到 |
| 六SNR、两个seed独立正式评估 | 未开始 | 未知 | 未开始 | 未开始 | 独立测试、配对样本和置信区间材料未找到 |
| Overleaf主稿构建链 | 完成 | 不适用 | 完成（构建成功） | 不适用 | 有最新18页PDF；页面人工检查未知。主稿尚未按K8计划改写 |

## 运行、更新数与checkpoint

- 当前运行状态：训练**跳过**（本次没有训练进程；不是训练失败）；CPU链路smoke**完成**。本轮没有刷新GPU快照，最近一次记录为11:51 CET、0%利用率。
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
| DL/UL与噪声 | UE完美DL CSI；BS使用LS＋U-Net UL CSI；gamma同条件下候选布局共同绝对N0 | 链路用N0公式并按相干假设跑通；U-Net和经验噪声方差验收未知 |
| 能量 | 每UE总能量64；候选Ep={8,16,24}，key:shared={1:3,1:1} | 六个候选合计64；CPU smoke单个候选实际数据能量最大绝对误差7.63e-6；尚未验证集选优 |
| 数据隔离 | 固定独立验证/测试namespace；测试仅用于最终报告，不能参与选优 | K8实际namespace、样本重叠检查及偏差均未知 |
| 控制时序 | pilot后先估UL，再选择组员/数据能量并通知UE，随后发key和共享数据 | CPU smoke按pilot调用、LS估计、数据调用顺序执行；无控制消息开销模型；相干假设为UL在两阶段间不变 |

## 结果和材料状态

以下“未知”表示本次没有找到并核验满足新K8协议的材料，不代表结果为零或通过。

| 材料 | 状态 | 关键数值/偏差 | 所需证据 |
|---|---|---|---|
| LS、LS＋U-Net UL NMSE–SNR曲线 | 未开始 | 只有LS＋线性插值占位（非U-Net）；无NMSE数字；pilot候选/位置已配置 | 独立校准数据、CSV、CE checkpoint与metadata |
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
- 最近已推送commit：`72f6792`；本轮K8链路源码尚未提交（待本次选择性备份）。
- 当前未提交源码变更：新增/修改K8协议、配置和链路文件；`.gitignore`修改；删除`diagnose_cross_cdl_condition.py`、`old/diagnostic_eval.py`、`old/train_adaptive_8ue.py`、`old/train_csinet_baseline_8ue.py`、`old/train_swin_baseline_8ue.py`。旧删除与其他未跟踪文件不是K8实现，不纳入后续源码备份提交。
- 本轮记录：增加能量网格和CPU物理反馈链路；未改历史K4训练代码、未改Overleaf tex、未启动GPU或训练。新K8源码与状态材料将单独提交并推送。

最多4个关键证据文件：

1. `/home/r50037942/project/03_PRECODER/docs/TWC_1_0_K8_two_week_plan_20261103.md`
2. `/home/r50037942/project/03_PRECODER/configs/twc_v1_k8_protocol_config.py`
3. `/home/r50037942/project/03_PRECODER/utils/twc_v1_k8_link.py`
4. `/home/r50037942/project/03_PRECODER/preflight_twc_v1_k8_link.py`

## 问题与下一步

1. **K8现有训练网络仍缺失。**当前成熟分组codec和hybrid power配置为K=4；新原型只到UL软码字恢复。影响：尚不能训练K8 Swin codec或生成下行CSI重建结果。
2. **真实K8信道和U-Net未接入。**当前smoke使用合成CPU信道和LS线性插值。影响：不提供UL NMSE/CE性能或真实CDL链路证据。
3. **选择策略和数据隔离未完成。**六个能量候选已配置，但pilot gamma查表、UL条件下Ekey/EAS/G选择及train/validation/test namespace均未确定。
4. **Git工作区已有旧源码删除和其他未跟踪内容。**后续备份只选择K8文件，不能混入原有删除、Overleaf或历史数据。
5. **主稿虽可编译，警告尚未处理，视觉预览未验收。**这不阻挡当前CPU协议工作；本轮按要求未编辑论文。

下一项动作：接入可回放K8 CDL信道并实现计划中的LS＋U-Net CE；配置独立数据namespace与采样记录；再补K8 CSI编码/重建/预编码接口。CPU链路验收通过后，进入GPU前置验收前先重新检查占用，超过20%请示。08目录不读取、不写入。

## 更新记录

- **2026-11-04 11:51 CET：**首次建立状态页。直接盘点Git、配置源码、实验输出目录、训练进程、GPU和Overleaf构建产物；未启动训练。记录了K8计划与现有K4代码/历史证据的差别，以及当前未提交改动。
- **2026-11-04 12:05 CET：**完成首轮场景、Hybrid/FDMA预算算术和反馈SNR点对齐；新建`docs/TWC_V1_PROTOCOL_BASELINE.md`。实测/代码映射、能量候选、噪声、数据隔离仍未验收；未使用GPU。
- **2026-11-04 12:09 CET：**新增独立K8预算/映射及配置模块；CPU核对静态编译、G分组数、Hybrid/FDMA总RE和六点N0公式均通过。未改旧K4训练模块、未启动GPU；K8模型训练集成仍未开始。
- **2026-11-04 12:09 CET：**K8协议源码及对应计划/状态文档提交为`b1573e2`，状态页更新提交为`016e489`；只纳入两份K8源码和三份K8计划/状态文档，未纳入原有删除或Overleaf目录。推送`origin main`成功，远端回执`ee58686..016e489`。
- **2026-11-04 12:09 CET：**状态记录修正并推送，远端从`016e489`前进至`bcb52ef`；未提交的旧删除、Overleaf、清理和历史数据仍保留在工作区。
- **2026-11-04 14:20 CET：**配置Ep/Ekey/EAS六候选，新增分散pilot的LS插值、key/shared发送及MMSE软恢复；`conda run -n precoder python preflight_twc_v1_k8_link.py`以CPU验证G=1/2/4均通过。该smoke未测性能，不含U-Net、DL CSI重建和预编码。新源码待选择性提交/推送。
- 后续每个主要阶段完成或出现影响进度的问题时，追加带时间的状态变更、证据路径及下一动作；保留负结果和历史记录，不覆盖已报告事实。
