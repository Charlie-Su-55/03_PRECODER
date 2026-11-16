# 03代码目录整理

本次整理针对入口混杂问题，保留当前和历史模型的源码身份。它不修改任何训练
配方、科学核心、网络、随机地址、结果、权重、论文、人工图、08或网络认证配置。

## 审计与实际变更

清理前根目录有179个Python文件，其中77个以train／eval／diagnose开头：

| 分类 | 数量 |
|---|---:|
| 当前86文件核心直接保护的入口 | 19 |
| 其他配置或历史源码pin保护的入口 | 33 |
| 有运行调用、未直接pin的入口 | 14 |
| 未发现运行调用或源码pin的叶子入口 | 11 |

最后11个中，6个仍属于通用数字基线、跨CDL和不完美CSI评估，继续保留；其余5个
早期K8原型按原字节移动到 [archive/early_k8](../archive/early_k8/README.md)。
当前根目录因此减少5个Python文件。归档清单不是通配符删除规则。

当前86个科学核心文件和父版本77个文件仍保留原路径、原字节。旧源码可能被
新代码导入、被历史manifest的source_sha256引用，或者在严格加载旧checkpoint
时重新计算身份；“训练完成”不意味着这些源码可以移动。

归档脚本内部依赖原ROOT，归档用途为历史参考；需要复现时使用归档README
指定的历史worktree。旧状态文档中的命令作为当时的记录保留，不代表当前入口。
本次没有建立自动运行旧原型的dispatcher。

## 当前视图

[03_CURRENT.code-workspace](../03_CURRENT.code-workspace)只配置Explorer的
`files.exclude`，精确收起103个现存根目录历史入口。当前低SNR的run／train／
evaluate／publish四个入口及runtime／loss／results保持可见。

不使用通配符删除、不重写 `.gitignore` 或私有 `.vscode`，不额外设置 `search.exclude`、
Python解释器、终端目录、Git配置或代理。Explorer隐藏项也可能被默认搜索排除；
搜索历史源码时关闭搜索面板的“使用排除设置和忽略文件”，或用终端 `rg`。
用普通“打开文件夹”即可查看完整目录。
新增加的未知文件不会被这份精确清单自动隐藏。

## 新服务器同步

新机当前没有正在执行的03科学计算，可在PR合并后正常同步。Git会应用这五项
归档移动；若旧路径有本地修改或未跟踪冲突，保留文件并回传错误，不能用reset、
clean或手动rm绕过。

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate precoder
cd -P /home/r50037942/project/03_PRECODER
git status --short --branch
git pull --ff-only origin main
"$CONDA_PREFIX/bin/python" -B -S tools/check_03_layout.py
```

随后在VS Code远程窗口选择“文件 → 从文件打开工作区”，打开
`/home/r50037942/project/03_PRECODER/03_CURRENT.code-workspace`。

检查器只读，不移动或删除SSH文件。若旧根目录存在额外本地副本，检查器会报告并
保留它们；云端无法预先确认SSH未跟踪manifest或外部脚本对这些副本的依赖。

## 旧服务器

旧机的low-SNR实验仍在运行，继续使用当前检出的版本。建议等本批完成并回传
结果后，再同步这次目录整理，避免在一次运行中切换Git提交。无需停止当前任务。
虽然86个受保护文件的字节没有改变，运行期间仍保持来源记录稳定更容易核查。

## 验证

归档源字节应与原提交的五个Git blob一致；当前86文件的SHA256必须全部匹配；
当前入口不能进入隐藏清单；workspace只允许精确根文件名。只读检查器和CPU
测试会验证这些条件。当前模型的47项CPU检查用于确认原调用链仍可用，不代表
新的模型性能验证。云端没有连接SSH、加载真实权重或启动训练。

本次实际验收：标准库 `-B -S` 的6项目录检查全部通过（0.285秒）；隐藏CUDA后
原低SNR实验47项CPU检查全部通过（27.830秒）。只读检查器报告archive5、
protected86、hidden103、active4，无待处理根目录副本。Git将五项识别为100%字节
一致的重命名，当前科研配置的core及parent身份检查均通过。
