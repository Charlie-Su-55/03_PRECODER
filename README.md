# 03_PRECODER

当前主线：K8、固定反馈资源与每UE能量下的 Hybrid 反馈和多用户下行干扰管理。
当前实验是两条配对的低反馈SNR续训轨迹；FDMA、CE及CE误差表保持冻结。

## 当前使用的入口

| 用途 | 文件／命令 |
|---|---|
| 启动检查与tmux | `bash tools/start_twc_v1_k8_low_snr_old.sh` |
| 实验队列与进度 | [run_twc_v1_k8_low_snr.py](run_twc_v1_k8_low_snr.py) |
| 训练与严格恢复 | [train_twc_v1_k8_low_snr.py](train_twc_v1_k8_low_snr.py) |
| 四模型配对评估 | [evaluate_twc_v1_k8_low_snr.py](evaluate_twc_v1_k8_low_snr.py) |
| 显式结果发布 | [publish_twc_v1_k8_low_snr.py](publish_twc_v1_k8_low_snr.py) |
| 配方、链路、目标、统计 | [config](configs/twc_v1_k8_low_snr_config.py)、[runtime](low_snr_runtime.py)、[loss](twc_v1_k8_low_snr.py)、[results](twc_v1_k8_low_snr_results.py) |

完整实验约定见 [TWC_V1_K8_LOW_SNR.md](docs/TWC_V1_K8_LOW_SNR.md)。常用命令：

```bash
"$CONDA_PREFIX/bin/python" -B run_twc_v1_k8_low_snr.py --summary
"$CONDA_PREFIX/bin/python" -B publish_twc_v1_k8_low_snr.py --publish
```

发布只在完整结果验收后显式执行。不要重复启动已在旧服务器运行的同一批次。

## 用当前工作区查看项目

在VS Code远程窗口选择 **文件 → 从文件打开工作区**，打开仓库中的
[03_CURRENT.code-workspace](03_CURRENT.code-workspace)。当前四个入口会保留在侧栏，
103个历史入口会在这个工作区收起。其文件仍在磁盘和Git中，Python导入、源码
哈希、解释器及终端环境都不改变。普通“打开文件夹”仍可查看全部源码。
搜索历史源码时，可在搜索面板关闭“使用排除设置和忽略文件”，或在终端使用 `rg`。

这是独立的可选工作区文件，不写入用户私有的 `.vscode/settings.json`。

## 历史代码与归档

[archive/early_k8](archive/early_k8/README.md) 保存五个已被ICC-aligned主线替代的
早期K8原型。归档字节与提交 `31c1fb30e6e2995d847a02226a6cbf19ff86b5b0` 一致。
这些脚本依赖原根目录布局，归档副本用于历史阅读，复现时使用对应提交的目录结构。

其余旧训练／评估文件中，很多仍被当前86个科研核心文件、历史runner、测试或
checkpoint源码身份校验引用。它们属于兼容依赖，即使对应实验已结束也需要保留
原路径和原字节。通用基线、跨CDL与不完美CSI评估工具也继续保留。

整理清单、边界及服务器步骤见 [代码目录整理说明](docs/CODE_LAYOUT.md)。

只读检查归档、当前入口和86个兼容源码的完整性：

```bash
"$CONDA_PREFIX/bin/python" -B -S tools/check_03_layout.py
```
