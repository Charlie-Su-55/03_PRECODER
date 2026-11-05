# TWC 1.0 K8 首轮网络设计冻结

日期：2026-11-05  
状态：**首轮架构设计冻结；目标 BS decoder 尚未实现或运行验收。**

本文件冻结首轮 K8 实验的网络类型、张量接口、主要维度和训练边界，避免 Stage B 在不同 decoder 结构间启动。冻结指设计基线冻结；实现中的原型仍需按本规格改造并通过 CPU 验收。本文不构成性能证据，也不改变已完成的 CE checkpoint。

## 1. 首轮模型清单

首轮只有三个学习网络：

1. 公共 UL 信道估计器：pilot-only 残差 U-Net（Stage A，已有训练权重）。
2. UE 反馈编码器：K8 共享 Swin 编码器（与 BS decoder 联合训练）。
3. BS 反馈接收与 CSI 重建器：跨用户 Joint Transformer＋空间 cross-attention＋全频域 Swin decoder（Stage B/C 目标网络）。

资源控制和预编码不增加神经网络：有限能量/分组候选由 validation-only 确定查表策略选择；RZF 正则系数同样由验证集扫描后按公共 feedback SNR 查表。小型策略 MLP、可学习 RZF alpha、WMMSE 网络均不属于首轮模型。

## 2. 固定物理协议与共同边界

- `K=8, Nt=32, Nsc=256`；每用户每帧总能量64；总反馈预算256 RE，其中64 pilot RE。
- 每用户 key 维数4；Hybrid 的 `G={1,2,4}`，每组共享维数分别160、80、40；FDMA 每用户24个正交数据RE。
- feedback SNR训练范围0–25 dB，正式点0/5/10/15/20/25 dB；DL SNR固定25 dB。
- 同一 feedback SNR 使用共同绝对噪声 `N0=0.25×10^(-gamma/10)`；CE、物理反馈、接收恢复、RZF和指标使用统一定义。
- BS神经网络仅接收部署时可见信息。真实 `H_dl` 只用于外部损失/指标；真实 `H_ul`、真CE误差、发送前未经过物理链路的码字、无噪声接收信号和实际噪声样本功率不能作为BS网络输入。
- 固定采用 pilot-LS＋频域线性插值作为 CE 初值；Stage A U-Net 使用已完成的公共训练/校准产物，codec训练中冻结。

## 3. Stage A：pilot-only 残差 U-Net

**接口与数据路径：** 每个样本接收 `Y_rx[B,32,256]`、已知 pilots、仅由 pilot RE 构造的 LS＋线性插值初值 `H_initial[B,8,32,256]`、`Ep` 和噪声/feedback SNR。网络把非 pilot-only 的观测位置清零；data-only RE不得影响估计。

**结构冻结：** 对每UE共享权重。频率一维 U-Net，输入特征共134通道：pilot-only接收实虚部64、初始UL CSI实虚部64、已知pilot实虚部2、UE自有pilot mask与pilot并集mask 2、归一化feedback SNR 1、显式Ep条件1。通道宽度32/64/96，两级下采样、线性插值上采样、skip connection；输出64通道复CSI残差并加到初值。

**训练与状态：** 既定5,000步、`B=16`、AdamW，学习率`1e-3`，weight decay`1e-5`、梯度裁剪1.0；loss为mean per-link UL NMSE，覆盖六个gamma与Ep={8,16,24}。当前Stage A已完成5000/5000，best/latest step5000，独立校准18/18完成。此网络在codec训练/评估时冻结；不为每个codec arm重新训练。

## 4. UE Swin 反馈编码器

输入为UE本地完美DL CSI `[B,8,32,256]` 和公共G条件。每UE使用相同权重：全256频点 angular-delay 变换→实虚部 patch embedding（patch 4）→宽度128、4头、window 4、MLP ratio 4的4个现有非移位 Swin block→LayerNorm与空间平均池化。

G的embedding加到每UE pooled token。两个复数输出头分别产生key 4维与最大shared 160维；依G截取160/80/40维。训练期间UE参数可更新，FDMA与Hybrid分别单独训练，不共享训练结果或checkpoint。

FDMA使用同一 UE 编码器在G=1公共条件下产生最大码字，再按既定规则取前24个复坐标发射。它在BS侧标记为24个FDMA-data槽位，不伪造key/shared接收观测。该选择保持发射端参数结构一致，FDMA训练独立。

## 5. Stage B/C：目标 BS Joint Transformer＋空间 Swin

### 5.1 规范化符号槽位

每用户最多164个符号槽位，布局内活动mask显式传递：

- Hybrid：key槽位0–3；shared槽位4–`3+a`，其中`a=160/80/40`。
- FDMA：槽位0–23为FDMA-data，其余屏蔽。
- 每个活动槽带用户ID、符号种类、局部序号、物理RE坐标、组ID、可靠度及复软估计。Hybrid G=1/2/4的活动联合query数分别为1312/672/352；FDMA为192。

共享RE对组内每UE承载各自的shared码字坐标；软恢复输出按用户保留，组信息和相同物理RE坐标显式编码。slot builder严格依赖`HybridLayout/FDMALayout`映射，不允许用数组序号代替真实RE坐标。

### 5.2 接收观测 memory

构造256个物理RE memory token，宽度128。每个token融合该RE的 `Y_rx` 实虚部（64维）、8用户 `H_ul_est` 的实虚部（512维）、RE坐标、pilot/key/shared/FDMA角色与归属、从软恢复映射来的可靠度，以及公开feedback SNR/N0/能量条件。各信息分支投影到128维后相加并LayerNorm。BS仅用估计UL CSI，不接收真实信道误差真值。

### 5.3 Joint Transformer与修正

对活动符号查询，输入投影特征为：复软符号实虚部、可靠度、物理RE坐标、归一化槽位序号、8维UE one-hot、符号种类及最多4组的组身份编码。投影宽度128。

采用**全8用户联合**的两层 pre-LN Transformer；每层先做4头self-attention，再对256个observation memory token做4头cross-attention，最后接GELU FFN `128→256→128`。不按组屏蔽跨用户attention；组身份和共享RE/信道耦合特征进入query及attention bias，使网络能学习组内共享关系，同时保留跨组用户CSI联合重建所需的信息。attention bias由可见元数据构成：同用户、同组、同物理RE及估计UL归一化空间相关；各项采用可学习、按头缩放的系数。dropout=0。

Joint context经LayerNorm和复数线性head预测每个活动槽位的残差，残差head权重标准差0.001、bias为0；残差直接加至软符号估计，随后再次应用活动mask。禁止接收功率重归一化，避免消除真实能量/噪声条件。

### 5.4 空间 cross-attention与CSI输出

每UE把修正后复符号按槽位规范顺序序列化：Hybrid为key后接shared，FDMA为24个FDMA-data槽；补零并携带activity/layout条件，通过共享线性层扩展到该用户的512个128维空间token，即64×8 angular-delay patch网格。加入行、列和用户位置编码。

将8用户共2048个空间query对活动Joint context执行4头cross-attention，残差加入空间token；随后每UE通过宽度128、4头、window4、MLP ratio4的4个既有非移位 Swin block，LayerNorm、patch输出投影和完整256-tap inverse angular-delay transform，输出有限complex64 `[B,8,32,256]` 物理CSI。残差输出投影以标准差0.001初始化、bias为0。

对长attention序列使用query chunk=64；训练时对attention/Swin block启用non-reentrant activation checkpointing。此设置是实现基线，不等于已验证显存足够；首次CUDA验收必须实测峰值显存和速度后才能启动正式warmup。

### 5.5 训练阶段与模型比较

- Stage B：FDMA和Hybrid分别训练；每个家族、每seed独立10,000步NMSE warmup。Hybrid一份条件化模型覆盖G=1/2/4及候选能量。两seed共4份warmup。
- Stage C：每份warmup同权重分叉为NMSE与联合loss两支，各20,000步；联合loss前5,000步渐增rate权重、后15,000步固定。四类结果各两个seed。架构完全相同，唯一有意差异是预先声明的目标函数。
- 不用旧原型step250 checkpoint初始化目标网络。该checkpoint参数结构对应不同的多源融合原型，仅保留调试。
- decoder、UE encoder及优化器/checkpoint严格版本化；更改网络层数、宽度、attention范围、slot映射或初始化都视为新架构版本，重新开始对应Stage B预算。

## 6. 非神经控制器与预编码器

资源控制器冻结为有限候选验证查表，不训练网络。`Ep∈{8,16,24}`只依据发送pilot前已知的feedback SNR/长期条件选择；pilot后分组、组员和`Ekey:Eshared∈{1:3,1:1}`才可依据`H_ul_est`及校准误差统计选取。正式测试前冻结validation选择结果。控制器不能读取真UL误差、真DL、test速率或真噪声样本功率。

预编码固定为全8用户联合RZF，按每个子载波总功率1归一化。正则系数通过冻结模型的validation扫描选取，部署时按公开feedback SNR查表；首轮不训练alpha网络。WMMSE只做冻结CSI/预编码的数学诊断，不作为待训练网络。

## 7. 目标架构验收门槛

目标BS网络实现后、任何Stage B训练前，必须完成：

1. Hybrid G=1/2/4及FDMA的活动槽位数量、资源映射、组mask和padding检查；查询数应为1312/672/352/192，memory为256，空间query为2048。
2. hooks证明观测memory、两层joint self/cross attention、软符号残差、空间cross-attention和四个Swin block均实际运行。
3. 输入边界审计排除真实CSI、真实码字、无噪声Y、真CE误差和测试信息；data-only Y按物理接收链路正常进入observation memory。
4. CPU前向与反向在所有布局下有限；NMSE loss对UE编码器、Joint层、残差头、空间attention和Swin输出头均产生有限非零梯度。padding query/key不产生输出或attention泄漏。
5. 输出`[B,8,32,256]`有限CSI，并通过现有联合RZF、功率归一化及完整物理链路接口。
6. 代码静态检查和上述CPU验收通过后，确认GPU可用，再运行首次CUDA显存/吞吐验收。此验收前不运行正式Stage B。

## 8. 冻结范围与已知限制

冻结范围是首轮K8设计，包括CE U-Net、UE Swin、Joint Transformer＋空间 Swin、FDMA共用槽位接口、loss阶段及非神经控制方式。训练/validation/test namespace、能量候选选择、分组策略的具体查表结果和RZF alpha数值仍须validation选定，不属于网络结构决策；测试集只在配置冻结后评估。

现有K8源码只实现并验收了pilot U-Net、UE Swin及多源卷积条件融合Swin原型。目标Joint decoder在本文件中为冻结规格，尚未写入模型代码、尚未CPU验收、没有性能证据。`nvidia-smi`本轮无法连接驱动，GPU状态未知；本轮未启动GPU作业。08项目只读，Overleaf论文未改。
