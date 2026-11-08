# TWC 1.0 K8 科研进度

最后核查：2026-11-08（用户回传冻结RZF诊断完成；本轮准备RZF匹配的NMSE/joint续训，尚无新批结果）
依据：已有Stage A/B、schedule-screen、ordered-v2及task-loss历史记录，用户提供的冻结RZF诊断结论，以及本地源码检查。本地本轮未读取服务器权重或完整结果包；历史验收的实际边界保留。所有性能数值均为validation证据；独立test未读取。

## 当前交接快照（权威）

- **当前任务：** 独立RZF匹配训练代码与本地20项针对性测试完成、全部通过且0 failures/errors/skips；独立`-B -S`检查15项通过、5项真实Torch/NumPy依赖缺失跳过，5文件语法和三入口标准库dry-run全部通过，新结果根未创建。对`0e0b6e8`的Git diff确认旧probe `SOURCE_FILES`中的33个既有文件全部未改。固定FDMA alpha0.03、G4 alpha0.1用于新训练、beta2校准和完整monitor；不训练alpha、不改solver/归一化、encoder/decoder/CE或物理协议。新根为`runs/twc_v1_k8_rzf_matched_training_v1/`，尚无真实checkpoint校准、训练或评估结果。
- **动机（用户回传）：** 旧冻结诊断中FDMA两模型均选0.03、G4两模型均选0.1；调alpha后旧joint没有保住全六SNR均值优势。旧小monitor仍在改善，不能据此认定全网格收敛。本地没有重跑该结果；本轮用同源初始化和训练/评估RZF匹配来检验loss效果，不预设必胜。
- **来源必须匹配：** 新`fdma_nmse`与`fdma_joint`都从旧`fdma_nmse`末步6048/累计22096初始化；新`g4_nmse`与`g4_joint`都从旧`g4_nmse`初始化，绝不从旧joint接着训练。来源训练实现`596d2a4`、旧RZF诊断`0e0b6e8`及内部`b6dc7ed`/`8096ed9`分别保留；源state严格加载，旧optimizer/beta只作证据。
- **固定预算与新beta2：** 四臂seed20261107、batch4，各12096步，共48384步；fresh AdamW LR2e-4、weight decay1e-5、clip1、constant scheduler，CE冻结，训练地址22097–34192。两NMSE源在新`codec_train`500100–500111、Ep16、六gamma各两次、G4 ratio1:3做新梯度校准，合并24条norm记录得到共享beta2=`0.25×median(g_N)/median(g_R)`；无optimizer更新，756步ramp包含预算，record独立且不可覆盖。
- **monitor与末态验证：** 本地步3024/6048/9072/12096在固定64MU validation地址8960–8975监测六gamma、G4两ratio，恢复原RNG/mode，不选best；16份小JSON随恢复保留。小nn合成测试仅验真实CPU默认RNG正常/异常恢复，未验真实CUDA RNG或64MU GPU。大型权重只保留滚动latest，末态12096用小terminal JSON/SHA别名、不另复制大final。最后四新模型与两个冻结NMSE源统一比较布局固定alpha及新末态global-alpha诊断；新64MU alpha校准6144–6159锁定后，256MU验证7168–7231，全部sample/batch配对与负结果保留；test封存。
- **下一步唯一动作：** 代码和轻量检查完成并同步03后，用`run_twc_v1_k8_rzf_matched_training.py --run`一次执行来源→beta2→四臂108步后自动至12096→末态alpha选择/验证→结果包，回传完整`--summary`；中断显式`--run --resume`。详见`docs/TWC_V1_K8_RZF_MATCHED_TRAINING.md`。不把末态新alpha回写训练，不提前声称收益或收敛。

## 冻结RZF诊断代码交接（历史记录）

以下为旧RZF诊断代码交付时的记录；其中“尚无结果”是当时状态。其完成结论现由用户回传，当前匹配续训仍须在服务器核验来源与原证据。

- **当前任务：** 独立冻结RZF诊断，唯一变化为评估时传入原solver/功率归一化的alpha；不训练、不创建optimizer、不重做beta校准，不改codec/CE/物理链路。输出独立放在`runs/twc_v1_k8_rzf_probe_v1/`。本地19项针对性测试全部通过、0 failures/errors/skips；独立`-B -S`检查16项通过、3项真实Torch依赖缺失跳过；4文件语法和两个标准库dry-run通过，新结果根未创建。尚无真实checkpoint选定alpha或新validation结果。
- **已完成来源（用户回传）：** `fdma_nmse`、`fdma_joint`、`g4_nmse`、`g4_joint`均完成task-loss新增6048步、累计22096步。本轮冻结其四个规定末步checkpoint，来源实现`596d2a4`与内部`b6dc7ed`/`8096ed9`链分别保存；真实权重和完成证据由服务器严格核验，本地没有重新读取它们。
- **固定选择协议：** alpha参考0.01，预定grid=`[0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0]`。仅用64完整MU validation样本、地址4096–4111，每checkpoint选一个global alpha；六gamma等权，G4两ratio共同等权，平局取较小值。不可覆盖selection record锁定后，在独立256 MU validation、地址5120–5183比较fixed与selected；test封存，不按SNR/ratio/样本另选alpha。
- **诊断边界：** 每batch/model/condition只forward一次H_hat并复用各alpha，选择/最终验证分别576/2304次网络forward，无持久H_hat缓存或源权重副本；raw NMSE及channel norm不应随alpha变化。真实H_dl只作离线评分，不做逐样本尺度拟合；记录相对alpha但不改绝对alpha部署。全局缩放小合成张量检查按预定elementwise容差通过，不证明真实模型全幅度原因；fixed/selected下全部六点和sample/batch配对区间均保留。区间仅描述固定checkpoint和本次固定选择下的评估采样，不含多seed或alpha选择不确定性。
- **趋势与下一步：** 只读已有progress，分别展示训练NMSE/rate/beta与最后四个固定monitor点；原alpha0.01、gamma10/Ep16小验证集（G4为1:3）的趋势不能写成全SNR收敛，也不改用新alpha重算。服务器用`run_twc_v1_k8_rzf_probe.py --run`一次完成来源→锁定选择→独立验证→趋势→结果包，回传完整`--summary`；中断显式`--run --resume`。详见`docs/TWC_V1_K8_RZF_PROBE.md`。本轮真实源模型/CE CUDA未运行、尚无选定alpha；无新训练预算，不排除未来经授权的续训。

## 训练目标筛查代码交接（历史记录）

以下为task-loss代码交付时的记录；其中“未运行”是当时状态。其四臂完成现已由用户回传，当前冻结诊断仍须在服务器重新核验来源，不把旧代码检查当作本批运行结果。

- **当前任务：** K8训练目标受控筛查代码和本地轻量检查已完成，唯一实验变化为NMSE与校准后联合速率loss；不改encoder、decoder、CE、物理链路、归一化或RZF。新结果根为`runs/twc_v1_k8_task_loss_screen_v1/`，本地24项针对性测试全部通过、0 failures/errors/skips；独立`-B -S`检查21项通过、3项真实Torch依赖缺失跳过；5文件语法和三入口标准库dry-run通过，未创建新结果根目录。未运行真实权重校准、训练或模型评估。
- **已完成来源（用户回传）：** 有序窗口v2四臂均完成各3024步、累计16048步；本轮从`ordered_fdma`与`ordered_g4`的最终权重出发。来源实现版本为`b6dc7ed`，其内部schedule-screen来源版本`8096ed965027a92f6e0738c7653b4c3e75f71b8c`继续单独保存，不以新HEAD覆盖。本地未重新加载这些真实权重，不把完成事实写成新task-loss结果。
- **四臂与预算：** `fdma_nmse`、`fdma_joint`、`g4_nmse`、`g4_joint`，seed20261107、batch4、各新增6048步，共24192步；前108步包含在预算内。全部使用fresh AdamW、LR2e-4、weight decay1e-5、clip1、constant scheduler，CE冻结，RZF alpha0.01不变。训练地址接续16049–22096。
- **校准与验证：** 校准只用`codec_train`新地址500000–500011，两个布局共24条梯度记录，beta=`0.25×median(NMSE梯度范数)/median(未缩放和速率梯度范数)`，两joint臂共用；不执行optimizer更新，756步warmup计入预算。FP64仅用于梯度范数诊断累加，不改变模型/梯度计算dtype。完整校准record、ID和SHA进入checkpoint；未完成的校准attempt保留并拒绝自动重校准。最终只评估末步权重及两个冻结来源，validation新地址3072–3135、256完整MU样本、54条件；sample/batch配对均保留。test继续封存。
- **下一步唯一动作：** 服务器安全同步本批提交后，用`run_twc_v1_k8_task_loss_screen.py --run`一次执行来源核验→校准→四臂108步后自动至6048→共同validation→CSV/JSON及ZIP结果包；回传完整`--summary`。中断后显式`--run --resume`，不另行手动启动校准。细则见`docs/TWC_V1_K8_TASK_LOSS_SCREEN.md`。真实源加载、CUDA梯度/显存、校准、训练/恢复及验证均待服务器执行；当前不宣称新loss获得增益，也不以测试结果选权重或调beta。

## 有序窗口v2修复交接（历史记录）

以下为有序窗口v2准备时的交接快照；其中“尚未运行”描述当时状态。当前完成状态以上方用户回传为准，v1失败证据和本地检查记录继续保留。

- **阶段状态：** Stage A/B与用户报告的schedule screen完成记录保留。用户回传有序窗口v1真实checkpoint CUDA预检在`ordered_fdma`、gamma0的RZF比较失败；四臂训练均未开始。v2改为保留原始global-mean数值路径，加两个零初始化、无bias的3968维有序窗口差分残差；本地29项针对性轻量测试全部通过，0 failures/errors/skips，六文件语法检查及三入口`-B -S --dry-run`通过，未创建v2结果根目录。新增来源optimizer参数顺序回归检查仅排除两个明确新增权重来核验旧moment的ID/shape/dtype，不加载旧optimizer到新训练，也未执行optimizer.step。与`795a0cc`逐字典核对六组比较容差完全一致。v2服务器预检/训练/验证尚未运行。Stage C和test继续封存。
- **v1失败证据（用户回传）：** raw最大误差约`1.34e-7`至`5e-7`、归一化发送码字`8.28054e-7`、H_hat=`1.32419e-6`、RZF=`2.54094e-5`；原RZF容差未通过，DL NMSE/rate尚未验收。本地没有读取该服务器report；不能把前面字段的小误差写成完整迁移通过。v1输出与失败回执保留，不改写、不删除。
- **v2边界：** 新namespace为`twc_v1_k8_ordered_encoder_screen_v2`，不沿用v1通过状态。使用原tokens的`mean(dim=1)`计算g，norm/Swin只运行一次，原W0/b完整保留；新增两矩阵仅作用于`p_j-p_32`，零初始化，无新bias。小张量FP32/FP64零残差两head均逐位相同，最大绝对/相对误差0；完整模型真实CUDA等价性尚未验收。参数增量仍为1,301,504；实际总参数及有限非零梯度待服务器核验。预检将收集全部字段并单列同H_hat重复RZF检查；不放宽原容差、不改RZF。
- **schedule-screen回传：** 能量比1:3、六SNR平均速率为分块Universal G4=`10.70676`、交错Universal G4=`13.47553`、G4 specialist=`15.70854`、同预算续训FDMA=`19.96734 bps/Hz`。这些为用户提供的validation结果，非本地独立复算；调度改善仍未消除Hybrid相对FDMA的差距。下一批同时给FDMA与G4模型同一种encoder改造，不据此预先归因或宣称优胜。
- **最新验证诊断：** `diagnose_twc_v1_k8_paired_link.py`在CPU上用16个validation样本、六个SNR、共同信道和共同pilot/data噪声，对四份10k checkpoint完成软符号恢复分解。key分支NMSE约0.0021–0.0024，FDMA data24约0.0025；shared为主要恢复瓶颈，跨六SNR平均NMSE：G=1约0.244–0.264、G=2约0.070–0.080、G=4约0.015–0.018。能量最大偏差低于1.1e-5。组内接收用户分量功率和不是MMSE后SINR。结论支持优先研究shared空间恢复和每个G的有效训练量，不能证明Universal权重共享是因果。
- **配对链路结果：** 四checkpoint共同validation评估完成，UL信道估计在各布局完全一致；该评估和本次诊断均为CPU、0次更新、`test_used=false`。详细结果位于`runs/twc_v1_k8_codec_warmup/paired_validation_cpu_20261106/`及`runs/twc_v1_k8_codec_warmup/paired_link_diagnostics_cpu_20261106_rerun1/`。首次诊断的`paired_link_diagnostics_cpu_20261106/`是不完整失败回执，正式结果只认`_rerun1/`。
- **模型文件位置：** 当前机器上的正式10k权重为`runs/twc_v1_k8_codec_warmup/{fdma,hybrid}/seed_2026110{6,7}_jointv1/checkpoint_latest.pt`，每个约289 MB，PyTorch CPU载入核验到对应arm/seed/step=10000，含optimizer与RNG状态。Hybrid seed 20261106的旧本地step500中断权重另存于`hybrid/seed_20261106_jointv1_local_step500_interrupted/`，不是正式权重。`runs/`和权重扩展名被`.gitignore`排除；仓库目前不含这些模型。尝试创建`stageb-warmup-2026-11-06` GitHub release并附上四个warmup checkpoint、CE checkpoint、校准表和SHA256清单，但上传端返回HTTP 403，核验release/资产均未创建成功。此环境无Git LFS；因此**模型尚未通过GitHub交付**。权重仍在当前SSH工作区，需另行从SSH复制，或解决GitHub大文件上传授权/对象存储。不要把权重未上传误读为checkpoint缺失。
- **Git交接：** 仓库`https://github.com/Charlie-Su-55/03_PRECODER.git`，分支`main`。本轮v2修复以有序窗口v1提交`795a0cc`为代码基点；来源schedule-screen实验仍绑定`8096ed965027a92f6e0738c7653b4c3e75f71b8c`，不把新HEAD冒充来源版本。旧诊断脚本`f3a67bb`已包含。v2提交与来源实验commit分别记录，最终提交与远端核验在本轮交付中报告。原有人工绘图、旧源码删除、Overleaf和数据文件不纳入本次提交。
- **下一步：** 服务器安全同步v2修复提交，用`run_twc_v1_k8_ordered_encoder_screen.py --run`在独立v2目录重做来源/迁移验收；通过后才自动执行四臂各3024步及256样本共同validation。四臂配方、来源及训练/validation地址不变，Stage C和test继续封存。完成或失败后回传完整`--summary`；v1失败不能通过改旧manifest、改RZF或扩大容差掩盖。

## 一屏摘要

- **当前阶段：** 旧训练及冻结RZF诊断历史保留；用户回传旧global-alpha选择完成。本轮准备从两个旧NMSE源展开同源、RZF匹配的四臂12096步续训，尚无新批校准、训练或验证结果。test继续封存。
- **状态：** CE优化更新 **5000/5000，完成**；latest与best checkpoint均保存、载入并核验到step 5000。独立校准18/18条件完成。
- **GPU快照：** 训练历史快照见下方记录；本次CPU诊断未使用GPU。当前没有新的实时GPU快照，本页不声称GPU当前空闲。Hybrid正式warmup在独享H100完成。
- **代码版本：** 来源训练实现`596d2a4`与旧RZF诊断`0e0b6e8`分别记录；内部ordered-v2版本`b6dc7ed`和schedule-screen版本`8096ed965027a92f6e0738c7653b4c3e75f71b8c`保留，新匹配训练commit另记。仓库为`https://github.com/Charlie-Su-55/03_PRECODER.git`，分支`main`。
- **提交范围：** 本轮只提交RZF匹配训练明确白名单中的独立配置/入口、针对性测试和文档（含本页）。已有`paper/plot_specialist_envelope.py`人工修改和四个未跟踪PDF/PNG保持原样，不纳入提交；模型、旧训练器/配置、物理链路、历史结果、08和Overleaf不改。
- **代码/验收/证据区分：** 目标BS receiver已实现并通过CPU四布局前向/梯度验收；四份FDMA/Hybrid warmup均完成并核验。独立性能证据尚无，Stage C尚未开始。
- **训练步数/checkpoint：** CE更新数 **5000/5000**，latest/best step5000。Stage B四份FDMA/Hybrid warmup各10,000/10,000，累计40,000/40,000。四份checkpoint均CPU载入核验arm/seed/step=10000并含optimizer、CPU/CUDA RNG。seed2 elapsed/peak：FDMA 2224.04秒/1,263,862,272 bytes，Hybrid 2688.3秒/1,267,254,272 bytes。validation-only lambda诊断0次权重更新、未读test；Rref FDMA/Hybrid分别15.8401041和6.9979448。候选lambda尚未冻结。seed1 FDMA耗时1785.91秒、peak 1,214,382,592 bytes；Hybrid H100段2557秒、peak 1,266,140,160 bytes，前500步另用118.36秒。
- **训练曲线和性能：** 两个seed 20261106 warmup都为NMSE_warmup，LR=2e-4、NMSE权重1/rate权重0。Hybrid H100 step10000固定validation已按G=1/2/4及六gamma摘要（详见下一条）；匹配FDMA/Hybrid固定验证对照待生成。此前step500 CPU配对诊断（4 batches/16样本）仅为不等训练预算下的历史早期检查，细节见更新记录。另完成CPU配对验证：同一8个CPU validation信道样本、六gamma、Ep=16、相同噪声；FDMA checkpoint step10000对Hybrid step500。跨六gamma均值，FDMA DL NMSE/rate=0.00810/17.678，Hybrid G=1/2/4分别0.40249/1.547、0.38826/1.550、0.38819/1.581；配对差Hybrid−FDMA DL NMSE约+0.380至+0.394、rate约−16.10至−16.13 bit/s/Hz。样本bootstrap 95%区间见summary.json。因更新数不匹配（10000对500），这反映当前checkpoint差距，不能归因为布局/模型优劣，也不是独立测试证据。CPU数据ID=`f0dec803f6343c9fa5df1d4d122e92923351a7722abf1df2ffe5b6ce67145fbd`；未启用GPU。已完成FDMA在固定validation（4 batches/16样本）step10000的gamma 0/5/10/15/20/25 dB均值DL NMSE为0.00857/0.00774/0.00751/0.00740/0.00735/0.00734，和速率为15.6875/16.4570/16.9164/16.8415/16.8887/16.9782 bit/s/Hz。属于训练中validation，尚无匹配控制或独立性能证据，不作最终结论。此前CE loss为mean per-link UL NMSE，LR=`1e-3`，无速率loss阶段。最终固定validation跨18条件平均U-Net NMSE `0.02663`（-15.75 dB），匹配LS `0.06058`（-12.18 dB），差约3.57 dB；每条件512个user-link样本，best/latest均step5000。此曲线用于设计与选优，不是独立测试结论。Codec step250 checkpoint在16个CPU validation-role样本上六个gamma均完成核验；DL NMSE均值约`0.4937–0.4957`，sum-rate约`1.609–1.630 bit/s/Hz`。该CPU生成数据ID与CUDA训练manifest的validation数据ID不同；数值仅作链路早期诊断，不作为训练曲线或性能结论。95%区间只按4个batch均值用t分布计算，test未读。
- **Hybrid验证补充（step10000）：** seed 20261106/20261107训练中validation汇总及局限见本节后续对应记录。四份10k checkpoint和日志已同步到当前项目；不能把训练中validation当作独立测试证据。
- **Hybrid seed 20261107 validation（step10000）：** 72行、LR=`2e-4`、NMSE_warmup。按gamma 0/5/10/15/20/25 dB：G1 DL NMSE=`0.04420/0.03578/0.02947/0.02429/0.02114/0.01978`、rate=`6.0798/6.6214/7.3131/8.0486/8.4185/8.4650`；G2 NMSE=`0.03819/0.03126/0.02898/0.02781/0.02701/0.02658`、rate=`6.3119/6.5094/6.6816/6.8420/6.8211/6.8362`；G4 NMSE=`0.02616/0.02389/0.02288/0.02236/0.02173/0.02141`、rate=`7.4192/7.4918/7.5668/7.5912/7.5512/7.6332` bit/s/Hz。与seed2 FDMA step10000六SNR平均对照，FDMA=`0.00940/15.7767`；Hybrid G1=`0.02911/7.4911`、G2=`0.02997/6.6670`、G4=`0.02307/7.5422`（NMSE/rate）。此为4 validation batch点估计；逐臂噪声循环未保证严格逐样本噪声配对，无配对置信区间，不是test证据。该seed原始validation日志仍需回传。
- **资源与隔离验收：** 新decoder CPU链路每UE数据能量最大绝对偏差`7.6294e-6`，RZF每子载波功率最大偏差`2.98e-7`；data-only Y扰动使输出最大变化`2.32e-4`（缩小四布局检查）、`8.17e-4`（full-width G=4）。全宽结构CPU通过，但GPU峰值显存/吞吐未知。test未读取；六SNR正式评估未知。
- **论文：** 发现主稿`TWC_progress_updated_analog_superposition.tex`。最新构建PDF为`overleaf/build/TWC_progress_updated_analog_superposition.pdf`，2026-11-04 11:23:50 CET生成，18页；构建成功但有warning，详见下文。新增投稿尺寸的K8 encoder/decoder矢量图PDF/SVG和PNG预览，尚未插入Overleaf；未改tex。
- **主要待办：** 从旧NMSE FDMA/G4源分别匹配初始化新NMSE/joint两臂，使用布局固定RZF与新beta2，完成12096步/臂和四次完整网格monitor；末态再单独做新alpha诊断及256MU配对验证。不得从旧joint初始化或把评估新alpha回写训练。没有本批结果前不宣称loss收益或收敛。独立test和Overleaf继续不动。

## 阶段状态

统一状态仅使用：**未开始、运行中、完成、失败、跳过**。代码、运行验收、性能证据分别标注，避免把实现等同于实验结论。

| 工作项 | 状态 | 代码实现 | 运行验收 | 独立性能证据 | 当前依据/缺口 |
|---|---|---|---|---|---|
| K8场景、RE预算与SNR点对齐 | 完成 | 文档基线完成 | 算术核对完成；程序映射未开始 | 不适用 | 见`TWC_V1_PROTOCOL_BASELINE.md`；Hybrid预算每种G均256 RE，FDMA为64+8×24=256 RE |
| K8布局/预算配置模块 | 完成 | 完成（协议层） | CPU检查完成 | 不适用 | `utils/twc_v1_k8_protocol.py`与`configs/twc_v1_k8_protocol_config.py`；不是训练就绪实现 |
| Ep/Ekey/EAS候选网格 | 完成 | 六个正值候选已配置 | CPU约束检查完成 | 未开始 | Ep={8,16,24}；key:shared={1:3,1:1}；等验证集选优 |
| K8 CPU物理反馈链路 | 完成 | pilot、LS插值、数据发射、key/shared MMSE软恢复已实现 | `preflight_twc_v1_k8_link.py`通过，G=1/2/4 | 未开始 | 插值不是U-Net；不含CSI重建、预编码或性能结论 |
| K8 pilot-only U-Net | 完成 | 共享频率残差U-Net适配K8/64 pilot RE、03噪声口径；输入含gamma与显式Ep | Stage A 5000/5000训练及独立校准18/18完成；pilot-only隔离前向通过 | 有限validation设计证据 | 固定CE checkpoint供codec阶段使用；LS初值仅由pilot构造 |
| K8 UE编码器与BS接收器原型 | 完成（原型） | UE共享Swin key/shared编码；BS多源条件融合Swin把卷积侧特征加到码字token后做空间Swin处理 | CPU原型前向及端到端反向通过 | 未开始 | 类名为`K8MultiSourceFusionSwinReceiver`；不是目标Joint Transformer＋Swin decoder |
| K8 CDL分角色数据与重放入口 | 完成 | 新增CE训练/codec训练/CE校准/验证/测试独立seed role和样本地址指纹；复用Sionna RNG隔离 | Sionna 2.0.1 CPU生成五角色各1对K8 CDL；固定验证与训练step地址可复现；未发现角色间指纹碰撞 | 不适用 | 仅小样本地址/重放验收；完整训练checkpoint暂停恢复与大规模数据审计未知 |
| LS初值方法screen | 完成 | 新增归一化Gram ridge Fourier-delay LS(rank4/rank8)，保留pilot-LS线性插值 | 32帧、6 gamma×3 Ep固定validation screen；线性插值18/18条件胜出 | 有限validation设计证据 | 小样本只用于初值设计，不能当正式UL性能结果；首轮CE暂定固定线性插值 |
| 完整K8协议/候选冻结 | 运行中 | 几何、六点能量候选已配置 | CPU链路烟雾检查通过；gamma查表策略未选 | 未开始 | 数据namespace、pilot gamma查表和UL状态选优仍待验证集核定 |
| CPU配置、预算和数据隔离检查 | 运行中 | 资源/能量约束已配置 | 真实CDL分角色小样本、seed寻址重放和RNG隔离通过；训练/验证/校准已实跑 | 不适用 | 完整大规模namespace碰撞审计未知；test role尚未用于任何选优或训练 |
| Stage A：LS与LS＋U-Net UL CE预训练及校准 | 完成 | LS初值为pilot-LS＋线性插值；`train_twc_v1_k8_ce.py`只读pilot-only观测，H_UL只作监督目标 | 5000/5000更新；best/latest权重step5000载入核验；独立calibration 18条件通过 | 有限validation设计证据 | validation较同条件LS低约3.57 dB；不代表下行任务或独立测试性能 |
| 原型链路：物理反馈到RZF | 完成（原型代码/CPU验收） | CE冻结、校准软恢复、Hybrid/FDMA、多源条件融合Swin、DL NMSE/RZF和速率指标 | CPU原型真实CDL前向及反向梯度通过 | 未开始 | 仅gamma=10、Ep=16原型链路检查；无性能证据 |
| 首轮网络设计冻结 | 完成 | CE U-Net、UE Swin、Joint Transformer＋空间Swin、非神经查表控制与RZF范围已冻结 | 文档交叉核对完成；不等于代码验收 | 未开始 | `docs/TWC_V1_K8_NETWORK_DESIGN_FREEZE.md`；控制策略具体validation查表值仍待选择 |
| 目标BS decoder实现与CPU验收 | 完成（CPU范围） | `models/twc_v1_k8_joint_decoder.py`实现全用户Joint attention、观测memory、残差修正、空间attention及Swin输出 | Hybrid G=1/2/4和FDMA完整CPU链通过；full-width 128/4-block G=4前向与反向通过；输入敏感性/梯度检查通过 | 未开始 | CUDA资源/吞吐、batch-size验收未知；随机初始化smoke数值不是性能证据 |
| Stage B：FDMA/Hybrid NMSE warmup | 完成 | 10k入口已实现；resume RNG mapping修复已推送 | 两个seed的FDMA与Hybrid均完成10k；四份checkpoint已核验 | 有训练中validation；独立性能证据未开始 | 40k/40k；Stage C前需冻结lambda/Rref |
| Stage C：匹配NMSE/联合loss终段 | 未开始 | 分叉训练器及validation梯度尺度lambda诊断代码已实现；lambda和Rref不自动猜测 | py_compile与CLI帮助检查通过；无Stage C训练/恢复运行 | 未开始 | 必须先有匹配Stage B 10k checkpoint；lambda候选需验证集诊断后冻结；计划每分支20k |
| Stage D：资源/能量策略、RZF及WMMSE诊断 | 未开始 | 未知 | 未开始 | 未开始 | 计划先验证集查表/扫描；新协议策略与结果未找到 |
| 六SNR、两个seed独立正式评估 | 未开始 | 未知 | 未开始 | 未开始 | 独立测试、配对样本和置信区间材料未找到 |
| Overleaf主稿构建链 | 完成 | 不适用 | 完成（构建成功） | 不适用 | 有最新18页PDF；页面人工检查未知。主稿尚未按K8计划改写 |

## 运行、更新数与checkpoint

- 当前运行状态：Stage A CE训练和独立校准均**完成**；Stage B seed 20261106的FDMA与Hybrid各10,000步均**完成**；seed 20261107 FDMA 10,000步**完成**；当前没有可核验的正在运行训练。
- K8正式CE更新数：**5000/5000**。FDMA seed 20261106更新10,000/10,000；Hybrid seed 20261106更新10,000/10,000；FDMA seed 20261107更新10,000/10,000；Stage B已完成30,000/40,000。Hybrid seed 20261107尚未开始/核验。旧原型250步仍不计数。
- 来源checkpoint累计步数：两份Stage B run均从CE checkpoint step5000开始；FDMA run实际更新10,000步，Hybrid run另行计数，不把来源step混入本轮更新数。
- 最后保存且核验checkpoint：latest step5000；best step5000；checkpoint均由PyTorch载入并核验内部step。
- 训练日志步数与权重步数：manifest、JSONL训练日志和latest权重均完成5000更新；详细日志/checkpoint位于`runs/twc_v1_k8_ce/seed_20261105/`。
- 首500步与1k步实测耗时约18秒/37秒；完整5000步实测180.06秒，独立校准约13秒。训练CUDA峰值`502,798,848` bytes（约0.47 GiB）。训练后GPU快照0%、18 MiB。
- FDMA正式checkpoint `runs/twc_v1_k8_codec_warmup/fdma/seed_20261106_jointv1/checkpoint_latest.pt`载入核验step10000，优化器及RNG状态存在；Hybrid H100 checkpoint `~/03_PRECODER/runs/twc_v1_k8_codec_warmup/hybrid/seed_20261106_jointv1/checkpoint_latest.pt`由用户在H100用CPU map_location核验step10000，optimizer/RNG状态存在，尚未回传本机。旧原型checkpoint位于`.../fdma/seed_20261106/`，只保留调试。
- Stage C代码位于`train_twc_v1_k8_codec_stage_c.py`和`diagnose_twc_v1_k8_stage_c_lambda.py`，共用配置`configs/twc_v1_k8_codec_stage_c_config.py`。Stage C入口严格要求对应arm/seed的Stage B step10k checkpoint；NMSE/joint从同一warmup权重分叉，均重新初始化相同AdamW及constant LambdaLR；joint显式要求验证集确定的lambda_target与Rref，并按前5k更新线性升权。诊断器只用validation role，报告共享UE encoder patch层梯度尺度与候选，不自动选参、不读取test。代码未训练，lambda数值未选。
- CPU在该step250 checkpoint上对16个validation-role样本完成gamma 0/5/10/15/20/25 dB前向；DL NMSE均值0.4937–0.4957，sum-rate均值1.6088–1.6296 bit/s/Hz。4-batch 95% t区间仅作诊断。由于replay data identity把运行设备写入data_id，CPU评估data_id=`f0dec803f6343c9fa5df1d4d122e92923351a7722abf1df2ffe5b6ce67145fbd`，与CUDA训练validation data_id=`addaaee65bbbb48afdab4acc44a3f638430e7572c612479086e4440dd44b786d`不同，因此不是训练时同一固定validation信道集；test未读取，optimizer updates=0。证据：`runs/twc_v1_k8_codec_warmup/fdma/seed_20261106/cpu_validation_step250_16samples.json`。

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
- 最近已推送commit：`27402c3`（K8 BS receiver状态与验收回执）；receiver源码commit为`8f8fa82`。历史CE材料提交为`026aaaf`。
- 本轮receiver源码、CPU preflight及两份JSON证据已选择性提交并推送。工作区仍有既有`.gitignore`修改、5个旧源码删除、V2计划文件、Overleaf副本和历史数据等未纳入本轮备份。
- **2026-11-04 14:31 CET：** 选择性提交并推送K8候选能量网格、CPU反馈链路和preflight；commit `629ae85`，远端`main`从`72f6792`前进。`git diff --check`通过；原有旧文件删除、`.gitignore`和未跟踪目录未纳入。
- `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_LS_validation_screen_20261104.csv`及同名JSON记录32帧screen及数据指纹；该结果不是正式CE曲线。

最多4个关键证据文件：

1. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_CE_validation_curve_20261104.csv`
2. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_CE_calibration_20261104.csv`
3. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_JOINT_RECEIVER_CPU_PREFLIGHT_20261105_small.json`
4. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_JOINT_RECEIVER_CPU_PREFLIGHT_20261105_full.json`

## 问题与下一步

1. **固定G specialist对照待设计。** 本次诊断显示shared软恢复随G显著改善；需用匹配训练预算比较固定G specialist与Universal模型，之后再决定Stage C和lambda/Rref冻结方案。独立测试尚未开始。
2. **08接口不能原样移植。**08 U-Net固定K4/32 pilot RE、噪声参考0.125；03需要K8/64 pilot RE及0.25参考。已适配维数和噪声换算，但没有迁移08训练/校准材料，也未证明跨设置泛化。
3. **选择策略和数据审计未完成。**六个能量候选已配置、样本角色namespace已实现；pilot gamma查表、UL条件下Ekey/EAS/G选择及大规模重叠/暂停恢复审计仍未完成。
4. **Git工作区已有旧源码删除和其他未跟踪内容。**后续备份只选择K8文件，不能混入原有删除、Overleaf或历史数据。
5. **主稿虽可编译，警告尚未处理，视觉预览未验收。**这不阻挡当前CPU协议工作；本轮按要求未编辑论文。

下一项动作：设计固定G specialist validation ablation，匹配Universal模型的更新预算；依据结果再确定Stage C及lambda/Rref。测试集保持未触碰；08只读；Overleaf论文未修改。

## 更新记录

- **2026-11-06（H100结果同步到当前SSH项目）：** 用户将H100备份上传至`runs/_incoming_h100_stageb_20261106/`。CPU SHA256核对确认CE checkpoint与calibration.csv和SSH现有文件完全一致；无需覆盖。H100 FDMA seed 20261107、Hybrid seed 20261106/20261107 checkpoint均以CPU `torch.load`核验arm/seed/step=10000、optimizer状态存在，三个manifest均为completed 10000/10000。FDMA seed 20261107与Hybrid seed 20261107原先不存在于正式目录，已从incoming复制到`runs/twc_v1_k8_codec_warmup/{fdma,hybrid}/seed_20261107_jointv1/`。Hybrid seed 20261106正式目录原有SSH checkpoint是step500/interrupted；为保留负面/中断历史，已移动到`runs/twc_v1_k8_codec_warmup/hybrid/seed_20261106_jointv1_local_step500_interrupted/`，H100完整10k run放入原正式目录。incoming原件保留。现四份正式Stage B warmup均在当前SSH项目可读取，Stage B总计40,000/40,000；未调用GPU、未读test。下一步仍是严格共同验证信道/噪声的validation-only对照，再冻结Stage C的lambda/Rref；不能把warmup validation当独立性能证据。

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
- **2026-11-04 23:23 CET：**按用户允许扩大为小评估：step250 FDMA checkpoint在CPU validation-role的16样本（4 batches）上，对六个gamma计算DL NMSE与和速率；CPU全部有限，test未读取、optimizer更新0。4-batch 95% Student-t区间仅属小样本诊断，不作为性能结论。事后核对发现CPU与CUDA训练的replay `data_id`因协议纳入执行设备而不同；这些样本不是训练时同一固定validation集，已在本状态页标明。JSON位于`runs/twc_v1_k8_codec_warmup/fdma/seed_20261106/cpu_validation_step250_16samples.json`。
- **2026-11-04 23:50 CET：**按用户要求补Stage C代码，未训练。新增20k terminal branch trainer（NMSE/joint）、严格要求Stage B对应10k warmup checkpoint、共享训练/验证数据地址、前5k lambda线性渐增、后15k固定、相同fresh AdamW与constant LambdaLR、best/latest与resume状态；新增validation-only共享层梯度尺度诊断，需传入完整warmup checkpoint后才可运行lambda候选。`py_compile`和两个CLI `--help`通过；无optimizer更新、无GPU调用。lambda/Rref数值待验证集诊断，Stage C运行验收未开始。
- **2026-11-04 23:50 CET：**Stage C训练器、lambda梯度诊断、配置和状态页已选择性推送，commit `41e3abc`；未纳入原有源码删除、Overleaf或历史数据。Stage C仍未训练，待完整10k warmup及验证集lambda/Rref诊断后启动。
- **2026-11-05 00:00 CET：**根据源码结构复核，澄清原`K8JointFeedbackDecoder`只有卷积多源条件特征加和Swin块，尚无计划的跨用户self-attention、观测cross-attention、软码字残差修正或空间cross-attention。实现类改名为`K8MultiSourceFusionSwinReceiver`，旧类名保留兼容别名，module/state_dict参数层级未变；Stage B step250标记为原型试跑、不计正式warmup预算。未训练、未用GPU；目标decoder设计/CPU验收列为当前首要未完成项。
- **2026-11-05 00:02 CET：**接收器原型重命名、训练状态页与preflight报告更正已选择性提交并推送，commit `2cdbc18`。权重参数层级保持不变；没有运行训练或GPU任务。
- **2026-11-05（本轮）：**在检查V2计划、K8源码与03既有Joint decoder参考后，冻结首轮三个学习网络及非神经控制方式，规格见`docs/TWC_V1_K8_NETWORK_DESIGN_FREEZE.md`。确定全8用户联合注意力、Hybrid/FDMA统一符号槽位、256-token观测memory、软符号残差、2048空间query cross-attention、完整256-tap Swin；CE U-Net和UE Swin按现有已实现结构冻结。资源策略与RZF alpha采用validation查表，不增加MLP。`nvidia-smi`仍无法连接驱动，GPU占用未知；未启动GPU作业。目标decoder仍待编码/CPU验收；训练更新0、checkpoint未变、无性能证据；论文未改。
- **2026-11-05（本轮备份）：**冻结规格和状态页选择性提交并推送至GitHub `main`，文档提交`e58b8cc`，格式修订提交`66ddd98`，状态同步提交`1b60aa8`。未纳入既有`.gitignore`、旧文件删除、V2计划、Overleaf或历史数据改动。
- **2026-11-05（BS接收器CPU验收）：**实现`K8JointTransformerSwinReceiver`并接入`K8FeedbackChain`；统一Hybrid/FDMA符号槽位映射、256观测memory、两层全8用户pre-LN self/cross-attention、attention bias、软符号复残差、4096空间query cross-attention及全256-tap四block Swin输出。`preflight_twc_v1_k8_codec_link.py`在CPU、codec_train样本上通过Hybrid G=1/2/4及FDMA前向/反向；活动query数1312/672/352/192，目标模块hook均触发，UE encoder、Joint层、残差、空间attention与Swin decoder梯度均有限非零。冻结width128/four-block full-width Hybrid G=4前向/反向通过。data-only Y扰动输出差最大值：缩小四布局验收`2.3237e-4`，full-width G4 `8.1739e-4`。输入签名不含真CSI/发送码字/真误差。实际能量最大误差`7.6294e-6`，RZF功率偏差最高`2.98e-7`。DL NMSE/速率只为随机初始化链路smoke，不作为性能证据；训练更新0、test未读取、CUDA/显存/吞吐未知。源码和证据`8f8fa82`、状态页`87e7507`已推送；证据JSON为`docs/evidence/TWC_V1_K8_JOINT_RECEIVER_CPU_PREFLIGHT_20261105_small.json`与`docs/evidence/TWC_V1_K8_JOINT_RECEIVER_CPU_PREFLIGHT_20261105_full.json`。
- 后续每个主要阶段完成或出现影响进度的问题时，追加带时间的状态变更、证据路径及下一动作；保留负结果和历史记录，不覆盖已报告事实。

- **2026-11-06（Stage B正式warmup启动）：** 用户确认GPU空闲并授权约一小时任务。启动前`nvidia-smi`实测0%、18 MiB，只有GNOME进程；新架构FDMA seed 20261106，run ID `seed_20261106_jointv1`，代码commit `de3de67b815a6ee8c3371c75f6148899b3f13f71`，batch4、目标10,000更新，LR 2e-4、weight decay 1e-5、NMSE warmup（权重NMSE=1/rate=0），每250步保存与验证。到step500，console记录elapsed 94.29秒、估计余时1791.44秒，峰值CUDA分配1,214,382,592 bytes；最新checkpoint CPU载入核验step500、arm/seed及optimizer状态存在，并含CPU/CUDA RNG状态，可续训。manifest异步滞后显示step250，故以console和实际checkpoint为准。训练中validation step500（16样本/4 batches）DL NMSE和速率如一屏摘要；不含匹配baseline，不构成独立性能证据。GPU快照68%、2074 MiB，Python进程2048 MiB与本run相符。当前动作：继续本run至10,000步或约一小时边界，在已保存checkpoint处停；随后核验权重和状态。证据：`runs/twc_v1_k8_codec_warmup/fdma/seed_20261106_jointv1/manifest.json`、`training.console.log`、`checkpoint_latest.pt`。

- **2026-11-06（FDMA warmup完成、Hybrid配对run启动）：** FDMA seed 20261106_jointv1完成10,000/10,000更新，manifest status=completed，耗时1785.91秒；checkpoint PyTorch CPU载入核验step10000、arm=fdma、seed=20261106，AdamW与CPU/CUDA RNG状态存在。峰值CUDA分配1,214,382,592 bytes。固定validation step10000（4 batches/16样本）六gamma的DL NMSE均值0.00857/0.00774/0.00751/0.00740/0.00735/0.00734，和速率15.6875/16.4570/16.9164/16.8415/16.8887/16.9782 bit/s/Hz；无匹配baseline、未跑test，不是独立性能结论。FDMA完成后GPU快照0%、18 MiB；随后启动Hybrid seed 20261106_jointv1，batch4、同10k目标和配置，当前运行中。此阶段没有源码改动；Git代码commit仍de3de67。证据路径：`runs/twc_v1_k8_codec_warmup/fdma/seed_20261106_jointv1/manifest.json`、`training.console.log`、`checkpoint_latest.pt`；Hybrid对应证据在`runs/twc_v1_k8_codec_warmup/hybrid/seed_20261106_jointv1/`。

- **2026-11-06（Hybrid因GPU外部负载暂停；修复续训）：** Hybrid seed 20261106_jointv1已保存checkpoint step500；CPU载入核验arm/seed/step、AdamW与CPU/CUDA RNG状态。发现另一Python进程PID 3367392占约1108 MiB，整卡利用率一度65%；本run在checkpoint后SIGINT退出。由于manifest在step250时异步落盘，曾落后权重；已将manifest补记为`interrupted`、optimizer_updates=500，elapsed 118.36秒，注明暂停原因。尝试恢复时发现原代码`torch.load(map_location=device)`会把CPU RNG tensor映射至CUDA，续训在任何optimizer更新前失败；修复为CPU加载checkpoint并让`load_state_dict`迁移状态。源码修复commit `ab6ccd4`已推送GitHub main；`py_compile`通过。最后GPU快照3%、1134 MiB，属于外部PID 3367392，实时值此后未知；未继续启动训练。FDMA completed 10,000步，仍为正式Stage B首份完成run。Hybrid validation step500记录在`validation.jsonl`，不作性能结论。证据路径：`runs/twc_v1_k8_codec_warmup/hybrid/seed_20261106_jointv1/manifest.json`、`checkpoint_latest.pt`、`validation.jsonl`；修复源码`train_twc_v1_k8_codec_warmup.py`。

- **2026-11-06（FDMA与Hybrid轻量CPU配对验证）：** 用户要求快速比较并尽量不使用GPU；评估将`CUDA_VISIBLE_DEVICES`置空，`device=cpu`，未启动训练、未读test。固定同一2批×4样本CPU validation数据，在六个gamma=0/5/10/15/20/25 dB、Ep=16、DL SNR=25 dB下，给FDMA step10000和Hybrid step500 checkpoint使用相同样本和按条件重放的相同噪声；Hybrid测G=1/2/4。共192行sample-level结果。六SNR均值上FDMA NMSE/rate=0.00810/17.678 bit/s/Hz；Hybrid G1=0.40249/1.547、G2=0.38826/1.550、G4=0.38819/1.581。配对差Hybrid−FDMA分别为G1 NMSE+0.3944 (bootstrap95% [0.3409,0.4580])、速率−16.1309 ([-17.9232,-14.4293])；G2 NMSE+0.3802 ([0.3286,0.4414])、速率−16.1281 ([-17.9785,-14.4367])；G4 NMSE+0.3801 ([0.3284,0.4427])、速率−16.0975 ([-17.8854,-14.4074])。区间按8个样本先跨六gamma平均再bootstrap，仅探索性描述。因为checkpoint训练步数10000对500，不是匹配训练对照，差距不能说明布局固有优劣；验证数据是CPU validation role，与GPU训练run data_id不同，test未读取。详细证据：`runs/twc_v1_k8_codec_warmup/cpu_pair_validation_20261106/summary.json`、`per_sample.csv`。

- **2026-11-06（Stage B首seed完成）：** 用户在H100独享环境恢复Hybrid seed 20261106；远端manifest报告status=completed、optimizer_updates=10000、H100段elapsed=2557.0秒、peak CUDA allocation=1,266,140,160 bytes。用户在H100以`torch.load(map_location="cpu")`核验checkpoint为arm=hybrid/seed=20261106/step=10000，optimizer、CPU RNG、CUDA RNG均存在。此前step1–500在另一GPU用时118.36秒；跨设备累计训练计算约2675秒，不含暂停和数据传输。与先前FDMA seed 20261106的已核验10k run组成首seed配对warmup；Stage B累计20k/40k。Hybrid是一套通过group_count条件覆盖G=1/2/4的模型，不是三套独立权重；训练循环以每18更新一个完整gamma×Ep网格轮换G，10k更新分布G1=3340、G2=3330、G4=3330。H100训练后验证曲线尚未摘要到本机，匹配配对曲线及独立测试仍待完成。下一run计划用第二seed 20261107先训FDMA，再训Hybrid。H100 checkpoint/manifest/log目前只在`/home/jupyter-r50037942/03_PRECODER/runs/twc_v1_k8_codec_warmup/hybrid/seed_20261106_jointv1/`，需要后续同步回本机。

- **2026-11-06（Hybrid step10000 validation摘要）：** 用户只读远端`validation.jsonl`，step=10000、LR=2e-4、loss stage=NMSE_warmup、72行。三种G×六SNR的DL NMSE与下行和速率摘要见一屏摘要；仅为validation，无置信区间、无test、未进行额外训练。原始JSONL仍在H100远端，需同步回本机。下一项为第二seed FDMA warmup；启动前需用户提供H100 GPU0最新`nvidia-smi`快照，若占用≤20%即可启动。

- **2026-11-06（seed 20261106 FDMA/Hybrid验证集对照）：** 以两臂各10,000步的NMSE warmup固定validation汇总作探索性比较，均为Ep=16、六个feedback SNR点，Hybrid按G分别统计。跨六SNR平均：FDMA DL NMSE/rate=`0.00765/16.6282 bit/s/Hz`；Hybrid G=1=`0.02262/9.1686`、G=2=`0.02098/9.2707`、G=4=`0.02138/8.8184`。Hybrid−FDMA点差分别为G1 `+0.01497/-7.4596`、G2 `+0.01333/-7.3575`、G4 `+0.01373/-7.8098`（NMSE/rate）。结果显示这组验证点上Hybrid落后，作为负向验证结果保留。两臂同seed、同训练预算、同固定validation信道批次；现有逐臂验证噪声循环不保证所有条件逐样本完全相同，故不报告配对置信区间。样本量为4个validation batch，未读取test，不能作为独立性能结论或跨seed结论。正式解释前应核验逐条件能量/软恢复与输出尺度，并在第二seed配对后复查。

- **2026-11-06（第二seed FDMA warmup，step500与step2750检查）：** 用户报告H100 run `seed_20261107_jointv1`状态running，manifest optimizer_updates=500时，validation事件step500、LR=2e-4、NMSE_warmup、24行；batch0在gamma 0/5/10/15/20/25 dB的前6行DL NMSE为0.3813/0.3769/0.3753/0.3754/0.3756/0.3759，rate为1.5377/1.5471/1.5451/1.5524/1.5561/1.5557 bit/s/Hz。此前用户粘贴3.56–3.61 NMSE行的事件step未知。之后console最新validation事件到step2750：elapsed=613.2秒、estimated remaining=1616.6秒、peak CUDA allocated=1,263,862,272 bytes（约1.18 GiB），实测约0.223秒/更新，估算总时长约37分钟。checkpoint以250更新间隔保存；manifest的`latest_checkpoint_step`字段为空，完成时需直接CPU载入核验step、optimizer和RNG。现有数值是训练中validation，不是独立测试或收敛证据；训练继续，未中断。

- **2026-11-06（第二seed FDMA进程退出，待核验）：** 用户报告H100 shell显示FDMA `nohup` job `[1]+ Done`。随后解析`training.console.log`的简短Python命令因日志中有JSON整数行而抛出`AttributeError: 'int' object has no attribute 'get'`；该解析异常发生在训练进程退出后，不代表训练失败。最终manifest status、optimizer_updates和checkpoint step尚未读取；不得仅凭shell `Done`当作训练验收通过。现有首seed（20261106）FDMA/Hybrid各10k fixed-validation汇总的六SNR平均对照及限制见上条；需CPU核验第二seed FDMA，再启动/确认同seed Hybrid并进行噪声严格匹配的validation-only评估。

- **2026-11-06（第二seed FDMA验收完成）：** 用户随后读取manifest与CPU载入checkpoint，核实`status=completed`、optimizer_updates=`10000/10000`、checkpoint arm=`fdma`、seed=`20261107`、step=`10000`，optimizer状态存在。shell `Done`后的日志摘要脚本错误仅由有效JSON整数行造成，不影响训练。H100最终elapsed、峰值显存和checkpoint RNG状态尚未读取。Stage B累计更新增至30,000/40,000；下一项为seed 20261107 Hybrid 10,000步warmup。完成该配对后进行共同验证信道与共同噪声的validation-only比较，保留负结果，不读取test。

- **2026-11-06（overnight Stage-B/诊断流水线首次尝试失败）：** H100 PID `3650932`以exit code 1结束。FDMA seed 20261107梯度诊断已完成并生成`fdma/seed_20261107_jointv1/lambda_diagnostic.json`：Rref=15.8401041 bit/s/Hz，validation-only、test_used=false、optimizer_updates=0。用户报告Hybrid目录无manifest/checkpoint/诊断结果，说明Hybrid训练未完成；依据源码`train_twc_v1_k8_codec_warmup.py`会拒绝run目录内除`training.console.log`以外的文件，根因为包装日志`overnight_stageb_validation.log`被放进同一Hybrid run目录。该诊断JSON不含`median_gradient_match_lambda`顶层字段；需查看`candidate_lambdas`中multiplier=1.0对应值，不能把打印None误判成诊断失败。修复为将日志移到run目录外，再单独重启Hybrid训练及诊断。Stage C未启动。

- **2026-11-06（Hybrid seed 20261107重启）：** 用户将首次尝试的wrapper日志与training console日志移到warmup run目录外，确认run目录没有残留文件后，以H100后台Bash PID `3652108`启动Hybrid seed 20261107 10k warmup，随后串行运行同seed validation-only梯度诊断。新wrapper日志在`runs/twc_v1_k8_codec_warmup/overnight_hybrid_seed_20261107.log`，训练日志在Hybrid run目录。用户后续核查进程为`bash`且manifest报告running、optimizer_updates=750/10000；实际checkpoint步数、训练耗时/峰值及最终诊断待后续验收。无需每500步停训。

- **2026-11-06（Stage B四份warmup完成）：** 用户对seed 20261107 FDMA/Hybrid的manifest、CPU载入checkpoint、训练日志和诊断JSON完成验收。四份正式warmup共40,000/40,000更新；Hybrid seed2 step10000、elapsed=2688.3秒、peak CUDA allocated=1,267,254,272 bytes；FDMA seed2 elapsed=2224.04秒、peak=1,263,862,272 bytes。两臂均保存optimizer和CPU/CUDA RNG。validation-only梯度诊断均optimizer_updates=0且test_used=false，FDMA/Hybrid Rref与lambda候选见一屏摘要；候选未选择。两seed step10000固定validation都显示FDMA高于Hybrid（点估计），但现有validation噪声未保证严格逐样本配对，未计算配对区间；保留负结果，不解读为独立测试或最终性能。Stage C未开始。

- **2026-11-06（四checkpoint共同样本CPU验证及UL信道估计误差）：** 新增`evaluate_twc_v1_k8_paired_validation.py`并完成py_compile与CLI检查。回执位于`runs/twc_v1_k8_codec_warmup/paired_validation_cpu_20261106/`，状态completed、CPU推理、16个validation-role样本、六个正式SNR、test_used=false、optimizer_updates=0；data_id=`8917b212ca7d7424e8242c61825984b49a3224beb64f5d83551fd1c0a1456c23`。四个10k checkpoint、相同DL/UL样本及逐条件共同pilot/data noise均用于FDMA/Hybrid G=1/2/4、两seed、Hybrid能量比1:3和1:1。校验确认各布局传给BS的`H_ul_est`逐元素一致。Ep=16时gamma 0/5/10/15/20/25 dB的UL NMSE：LS=`0.12132/0.06078/0.04193/0.03597/0.03399/0.03344`，trained U-Net=`0.08592/0.02826/0.00989/0.00409/0.00220/0.00161`；Ep=8/24和全部六SNR的LS/U-Net原始结果亦已保存，18/18条件U-Net点估计优于LS。Ep=16、跨六SNR平均DL NMSE/rate：FDMA seed20261106=`0.00890/16.955`、seed20261107=`0.01084/15.862`；Hybrid完整各G/能量比结果见CSV。12种seed×G×能量组合的Hybrid−FDMA六SNR配对NMSE差95%样本bootstrap区间均大于0；这是validation上的负结果，保留但不作为独立性能结论。第一次GPU推理因本环境`torch.cuda.is_available()=False`未能启动；全部评估在CPU完成，未训练、未读test。第二次inference_mode重复运行四批均完成；因避免覆盖既有证据目录，未改写CSV。下一步：结合该负结果诊断Hybrid codec/分组/能量与loss设计，并在Stage C训练前冻结lambda/Rref；测试集保持封存。

- **2026-11-06（Stage-B反馈链路分支诊断）：** 新增`diagnose_twc_v1_k8_paired_link.py`，使用同一16个validation-role样本、六gamma及共同pilot/data noise，对四个step10000 checkpoint分解软符号恢复、接收分量功率和能量。完整回执位于`runs/twc_v1_k8_codec_warmup/paired_link_diagnostics_cpu_20261106_rerun1/`；CPU、test_used=false、optimizer_updates=0。Hybrid key分支跨六SNR平均NMSE约`0.0021–0.0024`，接近FDMA 24维数据软恢复NMSE约`0.0025`；shared分支是主瓶颈：G=1约`0.244–0.264`，G=2约`0.070–0.080`，G=4约`0.015–0.018`（随seed/能量比例略变）。gamma=0时shared NMSE约G1=`0.52–0.56`、G2=`0.25–0.28`、G4=`0.066–0.078`；gamma=25时约G1=`0.047–0.053`、G2=`0.0029–0.0033`、G4=`0.00047–0.00051`。Hybrid 1:3时key/shared能量=12/36，1:1为24/24；总能量及分支能量最大实测绝对偏差均低于`1.1e-5`，未发现能量归一化实现错误。G1 shared维度160、名义每维能量0.225，接收desired shared分量功率均值约0.225、同组其他用户分量功率和约1.574；G2约0.447对1.342；G4约0.896对0.896。该干扰统计是分用户分量功率的非相干和，不等价于MMSE后的SINR，但与shared软恢复误差随G改善方向一致。1:3与1:1的shared NMSE差异较小且随G/seed方向不一，不足以解释FDMA与Hybrid整体差距。结果支持优先调查shared分支的空间复用/恢复和每G有效训练预算；不能据此认定Universal权重共享是因果。第一次link诊断的计算完成但CSV写出因两布局字段不同失败，残留`paired_link_diagnostics_cpu_20261106/`为不完整失败回执；修正后在`_rerun1/`成功重跑。下一步：设计固定G specialist validation ablation并明确匹配更新预算，再决定是否开Stage C；不读取test。
