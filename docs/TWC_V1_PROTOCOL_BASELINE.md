# TWC 1.0 K8 首轮协议基线

状态：**完成（首轮几何、预算、能量候选配置和CPU反馈链路原型）**
记录时间：2026-11-04 14:20 CET
依据：`docs/TWC_1_0_K8_two_week_plan_20261103.md` 的固定预算表和首轮场景；能量点是待验证的有限候选，未由验证集选择策略。

## 固定场景

| 项目 | 首轮基线 |
|---|---:|
| 同时服务用户数 K | 8 |
| BS天线数 Nt | 32 |
| 源子载波数 Nsc | 256 |
| 每帧总反馈预算 B | 256 RE |
| 正交UL pilot | 每UE 8 RE，共64 RE |
| 每UE key维数 d | 4，共32 key RE |
| 每UE总帧能量 | 64（Ep + Ekey + EAS；候选见下表） |
| 下行SNR | 固定25 dB |
| 正式反馈SNR点 | gamma = 0、5、10、15、20、25 dB |
| 训练反馈SNR范围 | gamma在[0,25] dB内采样（计划值；代码执行尚未验收） |

这里的六个正式“SNR点”指反馈链路条件 gamma；下行物理信道评估SNR固定为25 dB。后续图表和CSV字段应分别标为`feedback_gamma_db`与`downlink_snr_db`，避免混淆。

## Hybrid布局和RE预算

每个组复用其共享坐标，因此共享预算按组数G计为G·a，而不是按组内用户数重复计数。各组人数相等，g=K/G。Hybrid预算为

`64 pilot RE + 8×d key RE + G×a shared RE = 256 RE`，

其中d=4，所以共享部分总是160 RE：`G×a=160`。

| 总组数 G | 每组人数 g | Pilot RE | Key RE | 每组共享维数 a | 每UE数据坐标 d+a | 总共享RE G·a | 预算核算 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 64 | 32 | 160 | 164 | 160 | 64+32+160=256 |
| 2 | 4 | 64 | 32 | 80 | 84 | 160 | 64+32+2×80=256 |
| 4 | 2 | 64 | 32 | 40 | 44 | 160 | 64+32+4×40=256 |

上述“每UE数据坐标 d+a”包含该UE的4个key坐标和所属组共享分支坐标；共享坐标在同组用户之间复用。每个方案均保留正key和正共享资源。该表按计划算术核对，尚未由K8发射映射代码验证实际RE mask或能量。

## 匹配FDMA控制

匹配FDMA使用相同的64个pilot RE，剩余192 RE平均分给8个用户，即每UE 24个独立数据坐标：`64 + 8×24 = 256 RE`。该基线没有Hybrid key分支；因此不能把Hybrid的32个key RE再加到FDMA预算中。

## 首轮能量候选

配置了一个小型验证集筛选网格：`Ep ∈ {8,16,24}`，`Ekey:EAS ∈ {1:3,1:1}`。每个UE的六种候选如下；每行合计64，pilot分支在8个pilot RE上均分，对应单pilot RE能量1/2/3。

| Ep | Ekey | EAS | 每pilot RE能量 | 候选说明 |
|---:|---:|---:|---:|---|
| 8 | 14 | 42 | 1 | key:shared=1:3 |
| 8 | 28 | 28 | 1 | key:shared=1:1 |
| 16 | 12 | 36 | 2 | key:shared=1:3 |
| 16 | 24 | 24 | 2 | key:shared=1:1 |
| 24 | 10 | 30 | 3 | key:shared=1:3 |
| 24 | 20 | 20 | 3 | key:shared=1:1 |

`Ep`只由pilot前已知的反馈gamma/验证集查表策略确定；`Ekey:EAS`允许在pilot估计后选。网格只是启动验证所需的筛选点，不代表最优配置；当前没有选定pilot gamma查表或UL状态控制策略。候选由`configs/twc_v1_k8_protocol_config.py`产生。

## 噪声和证据边界

计划规定每个反馈gamma下，各候选布局使用相同的绝对N0，并以每UE参考能量/B定义噪声口径；每UE总能量为64、B=256。K8链路实现`N0=(64/256)×10^(-gamma/10)`。CPU smoke在gamma=10 dB时使用N0=0.025；尚未作多次统计来验收经验噪声方差。

## CPU反馈链路原型

新增`utils/twc_v1_k8_link.py`：pilot位置按`q=32×pilot_index+4×UE_index`分散到全频带；pilot后对每UE做LS和复数线性插值；数据发射按每UE能量归一化，key使用专属RE，组内shared符号在同RE叠加；接收端用单用户标量MMSE恢复key，用组内复数MMSE恢复shared。pilot发送与数据发送是分开的调用，原型假定两阶段之间UL信道保持不变。线性插值仅用于链路跑通，计划中的LS＋U-Net尚未实现。

在`precoder`环境运行`python preflight_twc_v1_k8_link.py`通过。CPU smoke使用batch=2、gamma=10 dB，对G=1/2/4均通过有限值和形状检查，布局均占256 RE；每UE数据能量最大绝对误差为`7.63e-6`，噪声口径N0=0.025。该结果是实现连通性检查，不是NMSE、速率或独立性能证据。

此原型尚未实现UE Swin CSI编码、BS Joint/Swin CSI重建、下行预编码或控制消息成本；不能当作完整端到端TWC方案。训练/验证/测试隔离和CE误差校准仍未知。

## CPU配置实现

新增独立模块`utils/twc_v1_k8_protocol.py`和`configs/twc_v1_k8_protocol_config.py`，不修改历史K4协议。模块提供K8固定常量、G=1/2/4布局、全频带分散的独占pilot/key与组内共享RE映射、FDMA控制映射、分组枚举和共同N0公式；配置提供六个有限能量候选，`EnergySplit`校验每UE Ep/Ekey/EAS均为正且合计64。候选不代表最优分配；查表和UL状态选优策略、训练配方和数据集namespace仍待后续阶段定义。

CPU检查通过：Python静态编译和`preflight_twc_v1_k8_link.py`均通过；布局分组数为1/35/105，Hybrid各自占满256 RE，FDMA为每UE24个数据RE、总计256 RE；物理链路G=1/2/4均跑通，能量最大误差7.63e-6。gamma=10 dB时N0=0.025。未调用GPU；未验证K8模型前向、LS＋U-Net接口、训练或性能。

## 更新记录

- **2026-11-04 12:05 CET：**对齐K=8、Nt=32、Nsc=256、B=256、64 pilot RE、d=4、G=1/2/4与六个反馈gamma评估点；复核Hybrid预算和FDMA控制预算，明确下行SNR另固定为25 dB。未启动GPU、未改Overleaf论文。
- **2026-11-04 12:09 CET：**新增独立K8协议和配置模块并完成CPU布局/预算检查；旧K4代码未改。
- **2026-11-04 14:20 CET：**增加六点能量候选及CPU物理反馈链路原型；`preflight_twc_v1_k8_link.py`在G=1/2/4均跑通。尚无U-Net、CSI重建、预编码或性能证据；未使用GPU。
