# K8 Codec链路训练前检查

核查时间：2026-11-04 22:14 CET。目的：补齐Stage B之前的代码并提供训练前可审阅报告。**没有启动训练；没有optimizer更新；未改Overleaf论文；08未访问或修改。**

## 一屏进度

- **当前阶段：** Stage A CE已完成；Stage B链路/训练入口已实现，CPU端到端验收完成；warmup训练尚未开始，等用户检查本报告。
- **代码、运行、证据：** 接口与入口代码已实现；真实CDL CPU前向、DL NMSE/速率/RZF计算、loss反向及能量/功率验收通过；独立codec性能证据无。随机初始化权重只用于链路自检。
- **配置：** K=8、Nt=32、Nsc=256、256 RE、64 pilot RE、key d=4、Hybrid G=1/2/4、FDMA每UE 24 data RE。CPU验收点gamma=10 dB、Ep=16、N0=0.025、每UE总能量64。正式六点0/5/10/15/20/25 dB尚未对完整链路逐点验收。
- **seed与版本：** CPU样本来自`codec_train`独立角色，数据seed=`20261104`；固定CE来源seed=`20261105`、CE更新5000/5000。代码基线commit=`7705c0cc5a382d4ad242e8cd649ceeb1b03e4cbd`；本轮相关源码修改待选择性提交。codec seed尚未启动。
- **更新/checkpoint：** Stage B更新`0/4×10,000`；来源累计codec更新0；无codec checkpoint。CE checkpoint `checkpoint_latest.pt`载入并核验step 5000；该来源checkpoint路径见下方证据。
- **loss/对照：** 新入口暂实现Stage B NMSE warmup（NMSE权重1、速率权重0，AdamW LR=`2e-4`）；验证固定信道上记录DL NMSE与真实下行和速率，覆盖六个gamma。G为1/2/4的Hybrid各记录，FDMA单独记录。没有训练曲线或匹配性能结论。
- **实际验收数值：** RE映射总预算偏差0；每UE数据能量目标48的最大绝对偏差`7.6294e-6`；RZF每子载波总发射功率目标1的最大偏差`3.5763e-7`；4种布局均有有限且非零codec梯度。CPU无显存数据，GPU峰值显存、吞吐与剩余时间均为“未知”。
- **资源状态：** 2026-11-04 22:09 CET `nvidia-smi`显示GPU利用率0%、显存18/32607 MiB；只运行CPU preflight，无GPU训练。
- **最重要限制：** smoke里的随机初始化codec DL NMSE和sum-rate仅证明张量链路可计算，不表示模型有效，也不能与CE的UL validation证据混为一谈。测试role未读取。

## 已接通链路

`真实K8 CDL → UL pilot发射 → pilot LS初值 → 冻结CE U-Net → 校准误差方差 → Hybrid/FDMA物理数据发送 → 基于估计UL CSI的软恢复 → 多源条件融合Swin接收器原型（卷积侧特征与软码字token相加，再经空间Swin块）→ DL CSI重建 → 8用户联合RZF → DL NMSE与真实信道sum-rate指标`。

**架构边界：** 该 CPU 验收只覆盖可运行的多源条件融合原型，不代表目标 Joint Transformer＋Swin decoder 已完成。跨用户 self-attention、观测 cross-attention、软码字残差修正和空间 cross-attention 在当时及当前均未实现；正式 codec warmup 应等待目标 decoder 设计冻结。

UE编码器只从仿真器提供的DL信道生成符号。BS的重建器输入中不包含真DL；真DL只传给训练监督和指标函数。DL发射功率按子载波对8用户联合precoder归一化为1。软恢复使用独立`ce_calibration`的标量各向同性误差方差插值；这是接收器近似，不宣称完整误差协方差。

## CPU验收结果与边界

使用实际Sionna K8 CDL `codec_train`样本和已训练/冻结的CE step-5000权重，在Hybrid G=1/2/4以及FDMA上完成前向、指标与反向梯度检查。每种布局损失和梯度有限、梯度非零，RZF功率归一化最大误差`3.5763e-7`。Hybrid和FDMA每UE数据能量均按Ep=16计算为48，最大实现偏差`7.6294e-6`。完整数据及随机未训练网络指标见机器可读JSON；不得引用其中的NMSE/sum-rate作为实验效果。

噪声口径仍为复噪声`E|n|²=N0=0.25×10^(-gamma/10)`。此前已验收的CPU噪声抽样最大相对偏差0.281%；本次链路只用gamma=10。训练/验证/校准/test沿用分角色寻址和隔离机制；本次只读取`codec_train`样本及CE校准材料，没有读取test。

## 训练入口与尚未验收事项

新增`train_twc_v1_k8_codec_warmup.py`及配置，可分别运行`--arm fdma`或`--arm hybrid`、显式seed和run-dir；每步按地址重放CDL训练样本，Hybrid轮换G/能量条件；固定验证集记录DL NMSE与下行和速率、gamma、Ep、G、loss阶段和学习率；保存optimizer/model状态及数据协议并支持续训。命令行、配置导入、`py_compile`和工作区`git diff --check`通过。

尚未做的验收：真实optimizer.step CPU smoke及暂停恢复；GPU显存/吞吐；所有正式gamma、Ep及分组/能量组合的物理链路矩阵；CE校准误差近似与实际软恢复误差匹配；训练数据完整隔离审计；正式验证策略冻结和独立测试。尤其是warmup验证代码目前覆盖全部六gamma、Ep=16，并在Hybrid下覆盖G=1/2/4；它不是所有六个能量候选的选优矩阵。loss联合阶段、正式两seed和统计区间尚未实现/运行。

开始训练后，先刷新GPU状态；超过20%利用率先请示。首个500–1000步按实际耗时更新剩余时间和峰值显存，不在每500步停下等待人工确认。Stage B仅在用户检查此报告并同意开始后启动。

## 关键证据（不超过4项）

1. `/home/r50037942/project/03_PRECODER/docs/evidence/TWC_V1_K8_CODEC_LINK_PREFLIGHT_20261104.json`
2. `/home/r50037942/project/03_PRECODER/preflight_twc_v1_k8_codec_link.py`
3. `/home/r50037942/project/03_PRECODER/train_twc_v1_k8_codec_warmup.py`
4. `/home/r50037942/project/03_PRECODER/runs/twc_v1_k8_ce/seed_20261105/checkpoint_latest.pt`
