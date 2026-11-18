# CLAUDE.md — 03_PRECODER 云端工作站守则

本仓库是 Rong Tan 的 TWC 论文项目：K8、固定反馈资源与每 UE 能量下，Hybrid（key＋共享叠加）反馈相对 FDMA 的多用户下行干扰管理。
新 agent 先读 `docs/AGENT_HANDOFF_20261117.md`，再按需读 `docs/` 下对应实验文档。**仓库是公开的**。

## 分工与硬边界

- 云端（Claude）：读源码、改代码与文档、跑 CPU 小测试、审计服务器回传的报告、设计实验。
- 服务器（用户执行）：所有 GPU/CUDA/Sionna、正式训练、真实 checkpoint 加载。云端不连接 SSH，不启动正式训练，不加载生产权重。
- 新机是训练和综合评估主机，旧机是历史备份与辅助小实验。两台的 repo 路径相同：`/home/r50037942/project/03_PRECODER`，conda 环境名都是 `precoder`。
- 不修改：`overleaf/`、论文稿、`paper/` 下人工材料、人工绘图、08 参考项目、历史 run 目录。
- 不追加 FDMA 训练，FDMA 权重冻结。若某项改动需要重训 FDMA（例如换网络结构），先向用户说明并等明确授权。
- test 集封存：不采样、不调参、不选模型。
- 不把权重、runs、报告、私有网络地址、代理配置、token、域密码提交进 git。不用 `git add .`，只按文件名添加。
- 不要求用户提供 token 或密码，不关闭 TLS 校验，不改永久代理配置。
- 服务器到 GitHub 的大于约 1 MiB 的推送目前返回 403，原因未定。完整报告留在服务器上；结果改用 `tools/results_digest.py` 生成几十 KB 的标量摘要，推到 `results-digest/<namespace>/<export_id>` 分支。用户不发附件。

## 协议常量（不要擅自改）

- K=8 UE，Nt=32，Nsc=256；总反馈 256 RE ＝ 64 pilot ＋ 192 data。每 UE 能量 64 ＝ pilot 16 ＋ data 48。
- FDMA：每 UE 24 个独占复坐标。Hybrid G4：4 个两人组，每 UE 4 个独占 key ＋ 40 个组内共享坐标。
- 反馈 γ ∈ {0,5,10,15,20,25} dB，`N0 = 0.25·10^(−γ/10)`，0.25 是每 UE 每 RE 的平均能量。下行 SNR 固定 25 dB。
- 下行固定 RZF `alpha=.03`，每 tone 总功率 1。信道为 Sionna CDL-C，UL/DL 独立生成，每 UE 平均功率已归一化，8 个 UE 共用同一个 γ。
- CE：pilot-only（rank8 Fourier-LS 初值＋残差 U-Net，CE 网络有 γ 特征）。final 系列 UE 编码器有 γ 输入（zero-init FiLM＋逐 slot 增益），审计显示作用很小（≤0.015 bps/Hz）。

## 实验与证据纪律

- 每个实验独立 namespace、独立 `runs/<namespace>/`、独立 `research/*` 分支；训练末态固定，不按监控曲线选 best checkpoint。
- 不覆盖、不删除已有 run；不放宽 checkpoint／receipt／来源 SHA 校验；不改历史 manifest。
- 配对统计：先在同一个完整 MU 内对 γ 取平均，再算 sample／batch 区间。六个 γ 不是六倍独立样本。线性 NMSE 先平均再转 dB。
- 报告里区分三件事：代码已实现、真实环境验收、性能证据。终端回传的摘要在审计完整 CSV 之前一律标注“未审计”。
- 保留全部负结果，不删点、不挑 γ。与冻结 FDMA 的比较要写明训练预算不同（Hybrid 累计更新更多）。
- 不把尚未实现或未训练的想法写成已完成；不预设 Hybrid 必须获胜。

## 云端常用命令

仓库 `tests/` 不是包，从仓库根目录按文件名模式运行，不要一次跑全部 153 个测试文件：

```bash
python3 -B -m unittest discover -s tests -p 'test_twc_v1_k8_low_snr*.py'
python3 -B -S tools/check_03_layout.py        # 只读，检查归档和入口完整性
```

云端有 CPU 版 torch，没有 Sionna、CUDA 和真实权重。

## 和用户沟通

- 用中文，一步一步来；先说结论，再说依据。
- 命令块里不要放需要用户替换的占位符。需要用户填写的值在命令块外单独说明，且不能与可执行命令放在同一块。
- 提供给用户的服务器命令要能整段粘贴；只读检查与会修改状态的命令分开。
- 失败的操作不要原样让用户重试，先换一个变量或补充诊断信息。
