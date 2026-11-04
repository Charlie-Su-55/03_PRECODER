# TWC 1.0 两周工作计划 · 实验、论文撰写与远程 PDF 预览

初版日期为 2026-11-03，本次修订于 2026-11-04，面向 Rong Tan 的 03_PRECODER。按实际启动日起算第1–14天；若11月4日作为第1天，则11月17日形成首轮实验与导师审阅稿。GPU阶段按共享服务器的实际可预约时间平移。新K8协议和耗时都需要实测。

论文已经上传到SSH服务器的 `~/project/03_PRECODER/overleaf/`。该目录是本轮论文工作目录，完整保留其中主tex、bib、图片、样式、已有PDF和人工修改，并加入后续清理保护清单。实际主tex与编译引擎由SSH Codex读取当前文件后确认。本次只更新工作计划，服务器工具安装与论文编译由SSH Codex执行并验收。

## 1. 两周目标与主场景

首版采用 K=8、Nt=32、源子载波数256、总反馈预算 B=256 RE、每 UE 每帧能量64、UE 侧完美 DL CSI、BS 侧通过正交导频 LS＋U-Net 取得 UL CSI。DL SNR 暂固定25 dB，反馈发射参考 gamma 在0–25 dB内训练，正式报告0/5/10/15/20/25六点。物理噪声按固定的每UE参考能量/B定义，在同一gamma下保持候选布局共同的绝对N0。

K8 已能研究每个共享 RE 复用2、4、8路的取舍。沿用每UE8个正交pilot时，K8的导频占64 RE（25%），K16占128 RE（50%）。K16适合后续验证负载趋势，首轮K8可以给key和共享数据留下更多空间。

两周的交付为可复现原型、两个codec seed的布局×loss主对照、六点性能与配对统计、资源/能量选择和干扰解释，以及一份能在SSH持续编译、在本地VS Code远程预览的导师审阅稿。写作从第1天开始，与实验并行。K16、SIP、变化导频数量、复杂CE联合微调列入后续扩展。CE预训练本轮使用一个公共checkpoint，两个codec seed共用它，因此两seed反映codec训练重复，不代表CE预训练也独立重复。

## 2. 固定预算和配置范围

首版先固定总导频数64、每UE key维数d=4。dimension自适应首先由组数G改变共享坐标数来实现。以下d=4是实施起点，不是已验证最优。

| 总组数G | 每组人数g | 总pilot RE | 总key RE | 每组共享维数a | 每UE数据坐标数d+a |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 64 | 32 | 160 | 164 |
| 2 | 4 | 64 | 32 | 80 | 84 |
| 4 | 2 | 64 | 32 | 40 | 44 |

预算满足64+8d+Ga=256。同样导频预算下，独立FDMA对照每UE24个数据坐标。Hybrid始终保留正key与共享资源、正分支能量。单组8人方案也保留key。

功率变量分为Ep、Ekey、EAS，每UE三项和为64。先固定pilot位置和数量，允许pilot能量随公共gamma调整。key/shared的比例可依据UL估计后的状态选择。第一轮只设少量候选或有界连续范围，并在第1–2天写入配置。扩大key维数或pilot数量范围会增加训练与校准需求，需单独记预算。

UE共用Swin编码器并使用key/共享输出头。Hybrid模型在训练中覆盖三种G及计划内能量配置。BS采用按组组织的软恢复与Joint模块、逐UE空间Swin重建，最后对全部8个UE联合预编码。同一Hybrid checkpoint用不同配置评价，属于配置条件化模型的性能；该范围内各固定分配取得的直接训练更新数应记录。

## 3. 分组评分与物理决策时序

若每个UE的UL空间向量作为G_q的一列，则R_q=G_q^H G_q。它的对角项是信道增益，非对角项衡量通道耦合。用归一化平方相关

rho_ij = mean_q |h_i(q)^H h_j(q)|^2 / (||h_i(q)||^2 ||h_j(q)||^2)

评价候选共享RE上用户的相似度。先逐RE取模平方再平均，避免不同频率的相位抵消。它是分离难度评分，实际干扰还受符号功率、CE误差和接收合并器影响。

K8分成两组各4人时有35种无标签分组；分成四组各2人时有105种无标签配对。可在CPU上对这种简单评分枚举。预先固定组到频域池的映射，隔离成员选择。先在同一G内选相关性低的分组；跨G的选择同时考虑坐标数和最终任务收益，避免单靠相关性评分总偏好最小组。

接收前后顺序为：

1. 公共噪声水平或长期配置决定pilot能量，发送前置正交pilots。
2. LS＋U-Net得到当前UL估计，条件校准给出误差统计。
3. 控制器依据UL估计、误差与公共条件选择G、组员和数据能量。
4. 将映射与能量配置通知UE，随后发送key＋共享坐标。

研究块需明确前置pilot、控制反馈和数据的时序与信道相干假设。pilot和data若在同一时刻发送，则不能利用本块pilot事后决定本块data分组。第一轮写明控制信息成本如何处理，不宣称已完成标准NR流程或净吞吐量验证。

## 4. 网络和训练stage

四个大模块为“资源与能量控制器 → UE Swin编码 → BS联合恢复与物理CSI重建 → 数学预编码器”。

### Stage A · UL CE监督预训练

LS＋U-Net约5k更新，覆盖0–25 dB及计划采用的pilot能量范围。输入包括实际pilot功率、位置mask、N0和LS初值。训练后用独立数据校准UL误差。动态Ep时，旧ICC按单一固定Ep校准的v(gamma)需要扩展。主codec实验先冻结公共CE。当前SIP不进入此阶段。

### Stage B · Codec重建预训练

每个布局家族、每个codec seed先训练10k，以物理DL NMSE为目标。两个家族是独立FDMA与配置条件化Hybrid，共4份warmup。Hybrid覆盖G=1/2/4和计划内能量范围，组员可随机变化以避免记住固定索引。保存第10k完整状态作为两loss分支的共同起点。warmup不从已经做过rate微调的旧模型开始。

### Stage C · 匹配的目标函数对照

每份warmup分叉出两个20k终段。NMSE控制分支继续最小化NMSE；联合分支在最初5k终段更新内逐渐增加速率权重，后15k保持预定权重，始终保留NMSE项。不加cosine。

联合loss为 L=NMSE-lambda(t)·Rsum/Rref。Rref用固定参考量缩放，lambda先由共享层梯度尺度与少量验证候选确定，不能将未经比较的系数解释为固定百分比。两终段以同样方式处理optimizer/scheduler，建议统一新建20k终段调度器，并显式保留模型与功率先验的累计状态。

最终4类结果×2codec seed，共8个checkpoint：FDMA-NMSE、FDMA-联合loss、Hybrid-NMSE、Hybrid-联合loss。共享warmup后，实际codec更新数为4×10k+8×20k=200k。每项最终结果均经历10k+20k=30k更新。首轮主对照统一使用末步30k，验证集best另作辅助记录，选择规则在测试前固定。

布局×loss的干净比较采用预先固定的Hybrid参考配置，例如G=2、d=4及固定能量。另用相同条件化checkpoint作配置和控制策略评估，避免把loss变化与布局策略变化混为一个因素。

### Stage D · 资源、能量与正则的轻量自适应

优先采用预先固定的有限候选与验证集查表。pilot能量只依赖发导频前已知的gamma/长期条件；组员及数据分能可依赖导频后的UL估计。候选涉及同一个Hybrid checkpoint支持的配置，无需每种G再训练新模型。可逐步加入CE-aware软恢复评分，以小MLP拟合控制策略属于可选追加。

RZF先由冻结模型的验证集扫描确定alpha(gamma)，在线只读取公共条件。随后有明确收益时再训练很小的正值alpha模块。WMMSE先测1/3/5次迭代的小样本冻结诊断，记录真实速率收益与运行时间，采用同一重建CSI设计波束。需要重新前向或统一计时的大批量用GPU，小样本已有张量的数学比较也可在CPU进行。

## 5. 两周日历与GPU预约

历史锚点为旧K4训练20k、batch48、每500更新验证约85–90分钟。新K8具有新的Swin/CE/变组路径，吞吐尚未测过。规划按20k约2–4小时、30k约3–6小时估算，均假设RTX5090独占与batch48。首个500–1000更新窗口记录包含验证和存盘的实测时间后，重算其余排期；正式更新计入既定预算。

| 时间 | 实验与代码交付 | 同期论文撰写与检查 | GPU需求 | 预计GPU小时 |
|---|---|---|---|---:|
| 第1–2天 | 固定预算、时序、G候选、能量范围、数据协议和对照；接口及CPU检查 | 接收并保护overleaf，定位主稿，配置编译/预览，建立原始基线；审阅旧稿，拟定新目录、贡献证据表及符号表 | 无 | 0 |
| 第3天 | 必要CUDA验收、吞吐与恢复检查；CE 5k及独立校准 | 写System Model与资源/能量约束，说明前置pilots、控制信息、分组和数据时序，画资源布局图初版 | 有 | 1.5–4 |
| 第4–5天 | 四份10k NMSE warmup，按首seed检查决定执行顺序 | 写UE Swin、LS＋U-Net、软恢复、Joint＋Swin、物理CSI输出及RZF；核对方法图和实现接口 | 有 | 4–8 |
| 第6–8天 | 首seed四个20k终段，含NMSE控制与联合loss | 写NMSE→联合loss训练阶段和控制器；完成实验设置、对照定义与统计方法；建立结果图表骨架，验证数据明确标注 | 有 | 8–16 |
| 第9天 | 验证结果读取，固定组员/能量策略与alpha候选；小规模扫描 | 形成方法与设置齐全的可编译中期稿；检查每条拟议贡献对应的实验，记录尚待正式验证的论断 | CPU为主＋GPU推理 | 1–3 |
| 第10–12天 | 第二seed同配方训练，统一独立测试 | 从正式CSV生成图表，写全部SNR及两seed结果；分析NMSE、信号、干扰、能量与分组的关系，保留负结果和适用范围 | 有 | 10–19 |
| 第13天 | 冻结模型alpha扫描、WMMSE少迭代诊断 | 写接收与预编码诊断、实际复杂度/耗时、限制及ICC差异；如实区分尚待验证的扩展 | CPU或GPU推理 | 1–3 |
| 第14天 | 结果、配置和版本冻结 | 按已验证结果重写Introduction贡献、Abstract和Conclusion；检查引用/公式/图表/页码/warning，形成审阅PDF及中英文导师摘要 | 无 | 0 |

上述表格合计25.5–53小时，统筹时将核心按30–55 GPU小时预约，再留10–15小时处理新模型问题和必要复核，总预留约40–70小时。新增论文写作与LaTeX编译由CPU完成，GPU预约预算沿用原计划。另预留约16–24小时人工审阅、修改与图表核对，分散到两周内；首次TeX环境配置约0.5–1.5小时，软件包下载和权限处理视服务器条件另计。可分为6–8个约8小时独占GPU夜间窗口，并留一个机动窗口。共享GPU时的墙钟时间按实测速率增加。

若头几天GPU一直被占，可完成第1–2天设计、CPU实现、文档与既有数据审阅，将后续GPU阶段顺延。日期是工作安排，不是对可用GPU的假设。新功能一次做必要验收，验收通过的模块直接续跑，不要求每500步人工停止。

## 6. 正式评价和阶段决策

固定独立验证/测试namespace，两seed共用测试信道，逐seed报告。主曲线为六gamma点的下行和频谱效率，同时保存物理DL NMSE、逐用户速率、Sk/Ik、CE误差、软码字误差、能量和所选G。配对差值与跨gamma均值按完整MU信道实现计算，保留sample/batch区间。

主要问题依次为：

- 在固定参考配置下，联合loss相对NMSE控制是否改善速率且重建仍可用？
- G=1/2/4的维数与分离代价如何随gamma变化？
- 同一G内，基于UL相关/误差的组员选择比固定分组改善多少？
- 同一checkpoint下，功率选择和组数选择各自带来多少增益？
- 对匹配FDMA，最终资源/能量策略在全部六点的差值是多少？

若前一阶段训练或物理链条出现明确异常，先修复并记录证据，再继续第二seed。若只有某一gamma获益，则如实报告范围。结果决定是否进入K16扩展、SIP或更复杂资源策略。1.0完成的是首轮可讨论的证据链。

## 7. 论文内容迁移与证据管理

从上传的journal主稿增量修改。先识别标题、documentclass、input/include关系及实际图片/bib依赖，建立一次原始上传快照或版本记录，再确定本轮唯一工作主稿。主稿可能仍叫TWC_progress_updated_analog_superposition.tex，须以实际文件为准。

写作围绕“固定反馈资源和每UE能量下的分组、维数及能量选择，最终怎样影响下行干扰与和速率”。方法章按K8新协议重写，明确64个pilot RE、每UE key4、G=1/2/4和可获得的信息。说明UE侧DL完美、BS侧UL来自LS＋U-Net，以及UL/DL独立快照假设。与已投稿ICC的关系在Related Work/贡献范围中交代，突出任务目标、资源控制和预编码的新增研究。

先写贡献候选与对应证据，正式结果完成后再收敛Abstract/Introduction/Conclusion措辞。维护“论断—图表/CSV—配置与checkpoint—状态”的小型证据表。旧Full/Lite、原接收SNR协议和新绝对噪声/能量协议的结果各自保留来源，只有协议一致的数据进入同一比较。新方案尚未完成的结果使用明确待验证标记；图表数值由CSV生成，验证集图在标题或说明中标出。

正文按用户偏好连续排版，只在段落、公式和环境边界换行。每次有实质修改后编译，保存错误、缺失引用、缺失资源、overfull/underfull等warning记录。重要图表、长公式和最终审阅版要实际打开页面检查。编译成功说明构建链可用，期刊版式与内容检查另行逐项完成。目标交付为导师审阅稿。

## 8. SSH写作、编译与PDF预览

工作目录固定为 `/home/r50037942/project/03_PRECODER/overleaf/`。推荐在本地VS Code通过Remote–SSH打开 `/home/r50037942/project/03_PRECODER`，由服务器编译，使用LaTeX Workshop的内置PDF标签页预览。SSH上的Codex CLI负责编辑与调用编译命令，用户通过VS Code查看PDF与SyncTeX定位。

### 首次配置

1. SSH Codex读取现有主稿与latexmkrc、字体/包需求，检查latexmk、pdflatex/xelatex/lualatex、bibtex/biber、kpsewhich及pdfinfo。优先使用已有可用工具链，根据实际主稿确定引擎和参考文献后端。
2. 如果缺依赖，按服务器发行版和权限安装满足主稿的最小TeX发行版组件与latexmk。需要IEEEtran、算法、TikZ/PGFPlots等包时据实际报错补齐。用户级TeX发行版可作为无系统安装权限时的方案，先说明安装路径与磁盘开销，沿用现有Python训练环境。
3. 用户在Remote–SSH窗口的扩展面板安装 `James-Yu.latex-workshop` 到该SSH主机。Remote–SSH扩展由本地VS Code使用。已有`.vscode/settings.json`需要合并设置，保留其他工作区配置。
4. 先对上传主稿做一次基线编译，记录主tex、编译器版本、新PDF路径、页数、引用与warning，确认生成时间及实际页面。保留上传PDF作为原稿基线。
5. 配置自动构建、输出目录与SyncTeX，用户打开真实生成的PDF确认显示。CLI编译成功与用户界面预览分别验收。

### 自动构建建议

将输出目录设为主tex旁的`build/`，预览器设为`tab`，启用SyncTeX。自动构建使用`onFileChange`，因为Codex CLI直接改文件也需要触发刷新。可设置2–3秒的最短构建间隔。CLI批量修改后仍可主动运行一次构建以读取完整日志。

以下是需要合并到工作区配置的设置示例。编译工具recipe按现场确认的引擎配置。

```json
{
  "latex-workshop.latex.autoBuild.run": "onFileChange",
  "latex-workshop.latex.autoBuild.interval": 3000,
  "latex-workshop.latex.outDir": "%DIR%/build",
  "latex-workshop.view.pdf.viewer": "tab"
}
```

如主稿使用pdfLaTeX，在实际主tex所在目录执行下面形式的命令；`MAIN.tex`需要替换为已核实的文件名。

```bash
latexmk -pdf -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build MAIN.tex
```

LaTeX Workshop的recipe应使用相同引擎、参数和输出目录。源文件拆分时用正确root指向工作主稿。VS Code命令面板可调用“Build LaTeX project”（通常Ctrl+Alt+B）和“View LaTeX PDF”（通常Ctrl+Alt+V）。以`build/`里的新生成PDF为预览目标。若观看不到PDF，先检查Remote–SSH扩展安装位置、构建日志和实际输出路径，再排查查看器连接。

官方参考：

- https://code.visualstudio.com/docs/remote/ssh
- https://github.com/James-Yu/LaTeX-Workshop/wiki/Remote
- https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile
- https://github.com/James-Yu/LaTeX-Workshop/wiki/View

## 9. 每日交付与保护范围

GPU实验和结果读取由SSH Codex在现有precoder环境完成，TeX编译使用独立的TeX工具链。GPU忙时可以继续论文审阅、编译、示意图和CSV处理，控制CPU构建并发即可。

每天保留一次可编译的进度稿、当日修改简述、实验状态、已验证结论与待办。图表、表格和关键数字关联小型CSV/metadata，最终审阅PDF、源文件和结果摘要一起保存。源码/论文源/必要数据按白名单做版本管理，忽略可再生TeX辅助文件与build产物；带日期的里程碑审阅PDF按需要单独保留。存在未知暂存内容时先核对，避免夹带提交。上传到服务器与同步Overleaf云端分别处理。

用户已在2026-11-04授权删除03旧权重和历史运行产物，旧恢复保护要求在该清理范围内撤销。实际是否已清完由清理回执确定，不能假定旧checkpoint仍存在。新K8实验独立命名。清理始终保护整个`overleaf/`、`paper/`、`paper_results/`、代码、Git历史及人工文件。08只供只读参考。

本次修订仅完成计划文档，服务器TeX环境、当前主稿、基线PDF编译和预览仍由SSH Codex现场核验。
