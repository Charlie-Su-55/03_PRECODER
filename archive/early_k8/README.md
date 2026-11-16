# 早期K8原型归档：仅供历史参考

这五个入口已经被ICC-aligned及后续联合训练主线替代。清理前核对了Python／shell
调用、测试、配置SOURCE_FILES／CORE_FILES和版本库内历史来源清单；没有发现
当前运行调用或源码pin，只有历史 [TWC_V1_STATUS.md](../../docs/TWC_V1_STATUS.md)
中的文字记录。服务器未跟踪的文件和外部调用不在云端审计范围内。

| 原根目录文件 | 归档用途 |
|---|---|
| `diagnose_twc_v1_k8_paired_link.py` | 早期配对链路检查 |
| `diagnose_twc_v1_k8_stage_c_lambda.py` | 早期Stage-C正则诊断 |
| `evaluate_twc_v1_k8_paired_validation.py` | 早期配对验证入口 |
| `train_twc_v1_k8_ce.py` | 早期K8 CE训练入口 |
| `train_twc_v1_k8_codec_stage_c.py` | 早期Stage-C codec训练入口 |

原路径、归档路径、SHA256和当前86个兼容文件的保护清单见
[manifest.json](manifest.json)。五个源文件按原字节保存，没有加入新ROOT逻辑或兼容shim。

## 复现原目录结构

归档文件内部使用 `ROOT = Path(__file__).resolve().parent`，因此**不要直接从
本归档目录执行**。原入口不再是当前支持的训练命令。需要复现旧原型时，先检查
当时的配置和依赖，在独立目录查看原提交：

```bash
git worktree add --detach ../03_PRECODER_early_k8_reference \
  31c1fb30e6e2995d847a02226a6cbf19ff86b5b0
```

该命令恢复原来的根目录路径，不启动训练。旧实验还需要当时的运行记录和权重；
不能把不同批次的checkpoint当成可互换来源。当前使用的入口见仓库
[README](../../README.md)。

论文、人工图、结果、日志及checkpoint均不属于本次整理对象。
