# AS 下一轮实验：服务器操作

本轮先补 K=4、Nsc=256、Dtot=256 的完整 baseline 对比。
该设置已有分配实验，优先复用服务器上的权重。
K=2、Nsc=128、Dtot=128 已准备好独立配置，作为下一轮。
本轮沿用现有网络、损失、反馈信道和功率归一化。

## 1. 同步代码，检查权重

将本次修改推送 GitHub，在服务器同步后，激活原训练环境并进入 03_PRECODER：

```bash
python run_as_screen.py check
```

此命令读取配置和路径，不导入 torch，不加载权重，不启动训练。
先把输出发回，以核实权重是否位于默认目录。

默认寻找：

- proposal：runs_deployzf_v1/checkpoints/
- 两个 baseline：runs_baselines/checkpoints/

如果你的 proposal 保存在 runs/，则使用：

```bash
python run_as_screen.py check --proposal-root runs
```

所有相对路径都以本脚本所在的 03_PRECODER 为起点。
多个候选权重不会按修改时间自动选取；使用输出提示的
--proposed-checkpoint、--swin-checkpoint 或 --csinet-checkpoint 指定。
缺少训练完成状态的历史权重会标为 unknown，需要核实来源。
不会自动使用已知暂停或失败的历史训练结果。

## 2. 确认路径后，在服务器运行

默认路径正确时：

```bash
python run_as_screen.py run
```

若检查时使用了路径参数，运行时也带上相同参数。

脚本依次执行：

- 检查 CUDA；无 CUDA 时退出。
- 对已有权重运行一个小批量的兼容性检查，保存在独立 preflight 子目录。
- 复用已有 best.pth；缺失模型使用现有训练入口训练。
- 新训练各 50,000 步，seed=42；支持从本轮 latest.pth 续训。
- 统一评估全部五种分配、Swin-CFNet 和 CsiNet+。
- 核对输出的场景、分配、样本数及数值有效性，生成 comparison.csv。

如果三个模型都缺失，会分别训练三个模型，总计 150,000 步；
这是最坏情况下的训练量，不是默认重新训练已有模型。
在服务器持久终端中运行；中断后重复相同 run 命令会复用或续训本轮模型。

新训练存放于 runs_as_screen/proposal/ 和 runs_as_screen/baselines/。
评估按时间戳另建目录，不覆盖原论文结果。

## 3. 返回结果

运行结束会显示结果目录。发回：

- comparison.csv：每种固定分配与两个 baseline 的平均速率差。
- summary_*.csv：完整结果及检查点路径、步数。
- screen_plan.json：本轮设置、复用/训练计划和执行命令。

samples_*.csv.gz 保留逐样本结果，用于后续配对分析。
preflight/ 只验证接口，其结果不用于论文比较。

评估统一设置：CDL-C，32 根 BS 天线，DL SNR=25 dB，
feedback SNR={0,5,10,15,20,25} dB，每个条件 1600 个 held-out samples。
使用同一批 DL/UL 信道评价各方法；不同张量尺寸下噪声不保证逐元素相同。
测试种子默认 20260921；这些结果不可直接混用旧图的样本均值。
所有分配点都保留，不输出按测试集挑选的“最佳分配”汇总行。

这是单个训练种子的筛查。平均速率的小幅领先不等于已验证的稳定优势；
有价值的设置随后用独立样本和重复训练复核，再决定论文结论。
历史权重的训练配方及网络复杂度未被此入口强制匹配。

## 备用：UE=2

等 K=4 结果分析完，再执行：

```bash
python run_as_screen.py check --case k2_n128_d128
python run_as_screen.py run --case k2_n128_d128
```

五种分配为 (64,0)、(48,32)、(32,64)、(16,96)、(0,128)。
proposal 的独立 suite 为 proposal_k2_screen，复用 K=4/Nsc=128 的训练配方；
两个 baseline 同样使用 50,000 步配置。原 proposal suite 的场景列表未增加 K=2。

本地仅进行了文本静态核对，没有运行 Python、训练或评估。
首次实际执行和兼容性验证由 GPU 服务器完成。
