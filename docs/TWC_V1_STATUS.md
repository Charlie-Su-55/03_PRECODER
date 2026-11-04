# TWC 1.0 K8 科研进度

最后核查：2026-11-04 23:50 CET
依据：K8 CPU链路验收、Stage B首次GPU运行日志/manifest/checkpoint、step250固定validation 16样本CPU评估、Stage C源码静态检查、Git状态和最近GPU快照。历史更新记录保留当时事实；小样本validation不是独立性能证据。

## 一屏摘要

- **当前阶段：** Stage A完成；Stage B因GPU占用暂停，FDMA seed 20261106为250/10,000；Stage C训练器与lambda诊断代码已实现，但Stage B warmup未完成，Stage C尚不能运行。
- **状态：** CE优化更新 **5000/5000，完成**；latest与best checkpoint均保存、载入并核验到step 5000。独立校准18/18条件完成。
- **GPU快照：** 2026-11-04 23:05 CET RTX 5090利用率68%、显存3538/32607 MiB；另一个项目Python PID 3175902占3512MiB。我们的训练进程已退出。
- **代码版本：** 最近已推送HEAD `468874a`；本轮Stage C代码和状态更新待选择性提交。GitHub origin为`https://github.com/Charlie-Su-55/03_PRECODER.git`。
- **未提交变更：** 本轮Stage C两份源码/配置/状态页尚未提交；原有`.gitignore`修改、5个旧源码删除、清理回执、SSH交接文档、Overleaf副本、历史数据及V2计划文件也未纳入本轮。
- **代码/验收/证据区分：** K8完整反馈链路、Stage B/Stage C训练入口及验证集lambda梯度诊断**代码已实现**；Stage B链路CPU前向和loss反向、Stage C静态编译与CLI参数解析**通过**；FDMA seed 20261106 Stage B运行过250步后暂停；Stage C运行验收和独立codec性能证据均**未开始**。
- **训练步数/checkpoint：** CE本轮更新数 **5000/5000**；latest与best均核验step5000。首个codec run本轮更新250/10,000、来源累计更新0；`fdma/seed_20261106/checkpoint_latest.pt`载入核验step250。其余三份warmup未开始。
- **训练曲线和性能：** CE loss为mean per-link UL NMSE，LR=`1e-3`，无速率loss阶段。最终固定validation跨18条件平均U-Net NMSE `0.02663`（-15.75 dB），匹配LS `0.06058`（-12.18 dB），差约3.57 dB；每条件512个user-link样本，best/latest均step5000。此曲线用于设计与选优，不是独立测试结论。Codec step250 checkpoint在16个固定validation样本上六个gamma均完成CPU核验；DL NMSE均值约`0.4937–0.4957`，sum-rate约`1.609–1.630 bit/s/Hz`。95%区间只按4个batch均值用t分布计算，属于早期小样本诊断，不作性能结论；test未读。
- **资源与隔离验收：** CPU链路能量最大偏差`7.6294e-6`、RZF功率最大偏差`3.5763e-7`。首个GPU warmup峰值显存`1,899,246,080` bytes（约1.77 GiB）；250步耗时31.28秒，按该初测估算当前run剩余9750步约20.3分钟。未达到500–1000步测速门槛，后续需要重算。test未读取；正式六点完整链路矩阵未知。
- **论文：** 发现主稿`TWC_progress_updated_analog_superposition.tex`。最新构建PDF为`overleaf/build/TWC_progress_updated_analog_superposition.pdf`，2026-11-04 11:23:50 CET生成，18页；构建成功但有warning，详见下文。该状态盘点没有改tex。
- **主要阻塞：** 外部项目进程占GPU约68%；FDMA seed 20261106已在保存的step250 checkpoint暂停，需等GPU空闲后恢复。启动日志目录检查问题已修复并推送；已保存更新未丢失。

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
| CPU配置、预算和数据隔离检查 | 运行中 | 资源/能量约束已配置 | 真实CDL分角色小样本、seed寻址重放和RNG隔离通过；训练/验证/校准已实跑 | 不适用 | 完整大规模namespace碰撞审计未知；test role尚未用于任何选优或训练 |
| Stage A：LS与LS＋U-Net UL CE预训练及校准 | 完成 | LS初值为pilot-LS＋线性插值；`train_twc_v1_k8_ce.py`只读pilot-only观测，H_UL只作监督目标 | 5000/5000更新；best/latest权重step5000载入核验；独立calibration 18条件通过 | 有限validation设计证据 | validation较同条件LS低约3.57 dB；不代表下行任务或独立测试性能 |
| 完整端到端CSI反馈到下行预编码 | 完成（代码/CPU验收） | CE冻结、校准软恢复、Hybrid/FDMA、Joint Swin、DL NMSE/RZF和速率指标代码已实现 | CPU真实CDL前向及反向梯度通过；GPU验收未开始 | 未开始 | `preflight_twc_v1_k8_codec_link.py`；仅gamma=10、Ep=16的随机初始化smoke，不是性能证据 |
| Stage B：FDMA/Hybrid NMSE warmup | 运行中（暂停） | 10k训练入口、固定验证DL NMSE/和速率、checkpoint/resume和manifest已实现 | FDMA seed 20261106更新250/10k，checkpoint step250核验；六点、16样本CPU validation通过；因外部GPU进程暂停 | 未开始 | 其余3份未开始；小样本为早期设计读数，独立codec性能证据无 |
| Stage C：匹配NMSE/联合loss终段 | 未开始 | 分叉训练器及validation梯度尺度lambda诊断代码已实现；lambda和Rref不自动猜测 | py_compile与CLI帮助检查通过；无Stage C训练/恢复运行 | 未开始 | 必须先有匹配Stage B 10k checkpoint；lambda候选需验证集诊断后冻结；计划每分支20k |
| Stage D：资源/能量策略、RZF及WMMSE诊断 | 未开始 | 未知 | 未开始 | 未开始 | 计划先验证集查表/扫描；新协议策略与结果未找到 |
| 六SNR、两个seed独立正式评估 | 未开始 | 未知 | 未开始 | 未开始 | 独立测试、配对样本和置信区间材料未找到 |
| Overleaf主稿构建链 | 完成 | 不适用 | 完成（构建成功） | 不适用 | 有最新18页PDF；页面人工检查未知。主稿尚未按K8计划改写 |

## 运行、更新数与checkpoint

- 当前运行状态：Stage A CE训练和自动独立校准均**完成**；本次codec tmux已退出，另有PID 3175902进程使用GPU。
- K8正式CE更新数：**5000/5000**；CPU smoke的3步没有并入正式训练预算。Stage B FDMA seed 20261106当前250/10k，4个warmup合计实际完成250步/40k计划更新。
- 来源checkpoint累计步数：0；CE从新初始化开始。
- 最后保存且核验checkpoint：latest step5000；best step5000；checkpoint均由PyTorch载入并核验内部step。
- 训练日志步数与权重步数：manifest、JSONL训练日志和latest权重均完成5000更新；详细日志/checkpoint位于`runs/twc_v1_k8_ce/seed_20261105/`。
- 首500步与1k步实测耗时约18秒/37秒；完整5000步实测180.06秒，独立校准约13秒。训练CUDA峰值`502,798,848` bytes（约0.47 GiB）。训练后GPU快照0%、18 MiB。
- Codec首run checkpoint位于`runs/twc_v1_k8_codec_warmup/fdma/seed_20261106/checkpoint_latest.pt`，step250可加载；运行时长31.28秒、峰值分配`1,899,246,080` bytes。以250步初测估算本run尚需约20.3分钟；未达到500–1000步要求，ETA为暂定值。
- Stage C代码位于`train_twc_v1_k8_codec_stage_c.py`和`diagnose_twc_v1_k8_stage_c_lambda.py`，共用配置`configs/twc_v1_k8_codec_stage_c_config.py`。Stage C入口严格要求对应arm/seed的Stage B step10k checkpoint；NMSE/joint从同一warmup权重分叉，均重新初始化相同AdamW及constant LambdaLR；joint显式要求验证集确定的lambda_target与Rref，并按前5k更新线性升权。诊断器只用validation role，报告共享UE encoder patch层梯度尺度与候选，不自动选参、不读取test。代码未训练，lambda数值未选。
- CPU在该step250 checkpoint上对16个固定validation样本完成gamma 0/5/10/15/20/25 dB前向；DL NMSE均值0.4937–0.4957，sum-rate均值1.6088–1.6296 bit/s/Hz。区间为4个batch均值的95% t区间，仅作早期诊断；test未读取，optimizer updates=0。证据：`runs/twc_v1_k8_codec_warmup/fdma/seed_20261106/cpu_validation_step250_16samples.json`。

## 配置与协议验收

| 项目 | K8计划值/规则 | 当前实际验收 |
|---|---|---|
| 用户数、天线、子载波 | K=8、Nt=32、Nsc=256 | 未开始；仓库中`grouped_codec_train_config.py`和`grouped_hybrid_power_train_config.py`的实验配置为K=4。旧`exp_8ue_128sc.py`是不同的128子载波旧方案，不能作本轮验收 |
| 反馈预算 | 256 RE；64 pilot RE；每UE key d=4 | CPU链路映射通过；总占用偏差0 RE |
| 分组预算 | G=1/2/4；每组共享维数a=160/80/40；预算64+8d+Ga=256 | CPU链路 smoke通过；RE偏差0；能量映射最大误差7.63e-6 |
| FDMA比较 | 不含key；64 pilot RE + 每UE24数据坐标×8 | 几何/预算映射CPU检查通过，偏差0 RE；FDMA无线链路尚未在smoke中执行 |
| SNR点 | 正式反馈gamma=0/5/10/15/20/25 dB；DL SNR固定25 dB | CE训练覆盖六个gamma与Ep全组合；DL链路未开始 |
| DL/UL与噪声 | UE完美DL CSI；BS使用LS＋U-Net UL CSI；N0=0.25×10^(-gamma/10) | CE训练真实CDL六gamma×Ep全覆盖；复噪声每点50万样本实测，最大相对方差偏差0.281%；DL任务链路未开始 |
| 能量 | 每UE总能量64；候选Ep={8,16,24}，key:shared={1:3,1:1} | 六个候选合计64；CPU smoke单个候选实际数据能量最大绝对误差7.63e-6；尚未验证集选优 |
| 数据隔离 | 独立CE训练、codec训练、CE校准、验证、测试namespace；测试仅用于最终报告 | 五个角色CPU样本地址各异，固定validation和ce_train step地址可重复；5对中跨角色SHA碰撞0。完整namespace规模审计、正式checkpoint恢复检查未知 |
| 控制时序 | pilot后先估UL，再选择组员/数据能量并通知UE，随后发key和共享数据 | CPU smoke按pilot调用、LS估计、数据调用顺序执行；无控制消息开销模型；相干假设为UL在两阶段间不变 |

## 结果和材料状态

以下“未知”表示本次没有找到并核验满足新K8协议的材料，不代表结果为零或通过。

| 材料 | 状态 | 关键数值/偏差 | 所需证据 |
|---|---|---|---|
| LS、LS＋U-Net UL NMSE–SNR曲线 | 完成（validation设计证据） | 18个gamma×Ep条件；最终跨条件U-Net NMSE 0.02663，LS 0.06058；每条件512 user-link样本。Ep=8/16/24，pilot总数64 RE | `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_CE_validation_curve_20261104.csv`和PNG/PDF；run ID `twc_v1_k8_ce_seed_20261105` |
| CE独立校准误差统计 | 完成 | 独立role 18条件×256帧；误差方差均有限 | `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_CE_calibration_20261104.csv`；报告记录hash |
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
- TWC K8论文材料：Stage A的UL NMSE验证曲线已有CSV/PDF及run ID `twc_v1_k8_ce_seed_20261105`；新协议下行NMSE/速率、FDMA配对和正式测试数字尚无CSV/run ID。旧稿里数字需逐项追溯，不能作为新协议性能证据。

## 源码、Git与证据路径

- 仓库：`/home/r50037942/project/03_PRECODER`；GitHub origin：`https://github.com/Charlie-Su-55/03_PRECODER.git`。
- 最近已推送commit：`026aaaf`（CE run报告、validation曲线与独立校准材料）；训练源码commit为`ec7a6fc`，其前一提交为`8342f7c`。
- 当前未提交源码变更：无；本轮CE报告、曲线、图和校准CSV已选择性提交推送。工作区仍有`.gitignore`修改及5个旧文件删除`diagnose_cross_cdl_condition.py`、`old/diagnostic_eval.py`、`old/train_adaptive_8ue.py`、`old/train_csinet_baseline_8ue.py`、`old/train_swin_baseline_8ue.py`。这些既有变更及其他未跟踪文件未纳入本轮备份提交。
- **2026-11-04 14:31 CET：** 选择性提交并推送K8候选能量网格、CPU反馈链路和preflight；commit `629ae85`，远端`main`从`72f6792`前进。`git diff --check`通过；原有旧文件删除、`.gitignore`和未跟踪目录未纳入。
- `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_LS_validation_screen_20261104.csv`及同名JSON记录32帧screen及数据指纹；该结果不是正式CE曲线。

最多4个关键证据文件：

1. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_CE_run_20261104.json`
2. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_CE_validation_curve_20261104.csv`
3. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_CE_validation_curve_20261104.pdf`
4. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_CE_calibration_20261104.csv`

## 问题与下一步

1. **K8 codec任务链路仍未打通。**CE trainer/calibrator已完成，但软恢复尚未读取UL误差校准并接入Swin，Joint CSI重建/RZF、FDMA匹配训练和评估入口未实现。影响：尚不能训练codec或生成K8下行重建/速率结果。
2. **08接口不能原样移植。**08 U-Net固定K4/32 pilot RE、噪声参考0.125；03需要K8/64 pilot RE及0.25参考。已适配维数和噪声换算，但没有迁移08训练/校准材料，也未证明跨设置泛化。
3. **选择策略和数据审计未完成。**六个能量候选已配置、样本角色namespace已实现；pilot gamma查表、UL条件下Ekey/EAS/G选择及大规模重叠/暂停恢复审计仍未完成。
4. **Git工作区已有旧源码删除和其他未跟踪内容。**后续备份只选择K8文件，不能混入原有删除、Overleaf或历史数据。
5. **主稿虽可编译，警告尚未处理，视觉预览未验收。**这不阻挡当前CPU协议工作；本轮按要求未编辑论文。

下一项动作：等GPU外部进程结束并确认利用率低于20%，再从FDMA seed 20261106 step250恢复，随后顺序跑完其余三个warmup。08保持只读；Overleaf论文未修改。

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
- **2026-11-04 17:13 CET：**按用户要求重跑CPU小任务，设置`CUDA_VISIBLE_DEVICES=`，无GPU训练/推理。`preflight_twc_v1_k8_data.py`通过：5个角色独立地址、交叉指纹碰撞0、validation固定重放与训练step重放一致、全局CPU RNG状态未改变；`preflight_twc_v1_k8_ce_chain.py`通过：2帧验证样本、6个gamma×3个Ep共18例，LS和未训练U-Net输出均为[B,8,32,256]且有限，优化更新0。第一次进程产生不可写Matplotlib默认缓存提示，之后CE检查将缓存重定向至`/tmp/mpl-twc03`且通过；该提示不影响检查。另核对现有评估入口：其规范和checkpoint仍属旧grouped-hybrid-power协议，K8正式评估/曲线产出代码尚未具备；需后续实现。未改源码和论文，未提交或推送；工作区原有修改保持不动。最近可见GPU快照为15:46 CET的76%，占用进程归属仍未知。
- **2026-11-04 21:20 CET：**新增K8 CE训练器`train_twc_v1_k8_ce.py`和冻结配置`configs/twc_v1_k8_ce_train_config.py`。按step循环覆盖6 gamma×3 Ep；每次固定验证复用相同信道，记录LS/U-Net NMSE、loss、梯度、学习率、耗时及峰值显存；支持原子best/latest checkpoint及optimizer/RNG续训；独立`ce_calibration` role输出18条件误差方差/NMSE。CPU smoke完成3次临时更新、18条件验证、续训和18条件独立校准；临时checkpoint不计正式更新。CUDA smoke启动检查失败：sandbox内`torch.cuda.is_available()==False`、device_count=0且`/dev/nvidia0`不存在；主机`nvidia-smi` 21:12 CET快照为GPU利用率0%、显存1124 MiB，存在约1098 MiB的未知计算进程。正式CE未启动。代码补齐与CPU验收估计约2–4小时（本阶段已完成）；正式CE约5k更新及校准按V2预留1.5–4 GPU小时，ETA和显存待在可访问GPU host测得首批500–1000更新后更新。代码待选择性提交/推送。
- **2026-11-04 21:40 CET：**已选择性提交并推送CE训练器和配置`8342f7c`。首次后台启动尝试在训练开始前安全退出：tmux将console log预先创建在run目录，脚本误将其判为已有训练产物。已修正为忽略这两个console log，`py_compile`通过；此修正尚未提交。失败尝试没有运行优化更新，最近GPU快照0%、18 MiB；正式CE仍未启动。
- **2026-11-04 21:43 CET：**修正后的启动目录保护提交并推送`ec7a6fc`。后台会话`twc03-ce-seed20261105`启动成功，Stage A CE正式训练至3500/5000更新；latest checkpoint step3500、best step3250均载入核验。固定validation最近值NMSE 0.02705，LS 0.06058；首500/1000步约18/37秒，训练剩余估算70–80秒，随后自动独立校准约0.5–1分钟；峰值CUDA分配502,798,848 bytes。训练时`nvidia-smi`最后一次可读快照48%、1406 MiB，之后无法从当前sandbox连接驱动，但GPU训练仍在tmux运行。所有性能值目前仅是验证集过程值。
- **2026-11-04 21:45 CET：**Stage A CE训练及校准完成。run ID=`twc_v1_k8_ce_seed_20261105`，新初始化seed 20261105，5000/5000更新，GPU训练180.06秒；latest/best checkpoint内部step均为5000且成功载入，峰值分配502,798,848 bytes。loss为mean per-link UL NMSE，LR=0.001，覆盖gamma 0/5/10/15/20/25 dB与Ep 8/16/24。固定validation 18条件平均U-Net NMSE 0.026625、匹配LS 0.060585（相差约3.57 dB）；每条件512 user-link样本，属验证设计证据。独立calibration role每条件256帧、18条件方差有限；不触碰test。CPU噪声采样每gamma 500,000复样本，经验E|n|²最大相对偏差0.281%。训练后GPU 0%、18 MiB。已生成验证曲线PNG/PDF和CSV、独立校准CSV及带checkpoint hash的JSON回执；论文未修改。下一步为完整codec接收/预编码链路。
- **2026-11-04 22:14 CET：**Stage B前置链路代码已实现并做CPU端到端验收：实际CDL codec_train样本经CE step5000权重与18条件校准、Hybrid G=1/2/4及FDMA物理发射/软恢复、Joint Swin重建、DL NMSE、真实信道下行和速率与RZF；4种布局均反向得到有限非零codec梯度，未执行optimizer.step，更新数0。Ep=16、gamma=10时Hybrid与FDMA per-UE总数据能量目标48，最大测得偏差`7.6294e-6`；RZF每子载波总功率目标1，最大偏差`3.5763e-7`。随机未训练网络的DL NMSE/速率值仅是实现smoke，明确不作为性能证据。新增Stage B FDMA/Hybrid 10k NMSE warmup入口；命令行、配置和py_compile通过，未真实训练/恢复。nvidia-smi 22:09 CET为0%、18MiB；没有使用GPU。GPU峰值显存、更新速度、剩余训练时间未知。报告见`docs/TWC_V1_CODEC_LINK_PREFLIGHT_20261104.md`。
- **2026-11-04 22:16 CET：**上述代码与报告选择性提交并推送到GitHub `main`，commit `4b73017e7f45b1f953459021de3a99b6741babbb`。原有`.gitignore`、5个旧文件删除及Overleaf/历史数据等工作区改动未纳入提交。训练继续保持未启动，待用户检查本报告。
- **2026-11-04 23:05 CET：**用户批准开始Stage B四份warmup。首次启动因run目录内预建的console log被误认作旧实验产物，在optimizer更新前退出；修复并推送commit `62c1d35`后重启。FDMA seed 20261106完成250/10,000更新，latest checkpoint载入核验step250；耗时31.28秒，峰值CUDA分配`1,899,246,080` bytes。启动时GPU空闲；其后发现另一个项目Python PID 3175902占用约3512MiB、GPU利用率68%。按GPU占用规则发送SIGINT，训练在已保存的step250 checkpoint后退出。其余三份warmup尚未开始；等待用户说明该进程是否属于其任务，或等待GPU空闲后恢复。
- **2026-11-04 23:13 CET：**应用户要求执行CPU小任务：step250 FDMA checkpoint在固定validation样本、六个gamma上完成推理与DL NMSE/下行和速率计算，结果全有限；没有optimizer更新、没有读取test、没有使用GPU。单样本值只用于早期validation诊断。JSON保存在run目录`cpu_validation_step250.json`。
- **2026-11-04 23:23 CET：**按用户允许扩大为小评估：step250 FDMA checkpoint在固定validation role的16个样本（4 batches）上，对六个gamma计算DL NMSE与和速率；CPU全部有限，test未读取、optimizer更新0。报告4个batch均值的95% Student-t区间，仅属小样本诊断，不作为性能结论。JSON位于`runs/twc_v1_k8_codec_warmup/fdma/seed_20261106/cpu_validation_step250_16samples.json`。
- **2026-11-04 23:50 CET：**按用户要求补Stage C代码，未训练。新增20k terminal branch trainer（NMSE/joint）、严格要求Stage B对应10k warmup checkpoint、共享训练/验证数据地址、前5k lambda线性渐增、后15k固定、相同fresh AdamW与constant LambdaLR、best/latest与resume状态；新增validation-only共享层梯度尺度诊断，需传入完整warmup checkpoint后才可运行lambda候选。`py_compile`和两个CLI `--help`通过；无optimizer更新、无GPU调用。lambda/Rref数值待验证集诊断，Stage C运行验收未开始。
- 后续每个主要阶段完成或出现影响进度的问题时，追加带时间的状态变更、证据路径及下一动作；保留负结果和历史记录，不覆盖已报告事实。
