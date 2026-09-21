"""
IEEE TWC Figure 3: Phase Transition across K=2, 4, 8
====================================================
展示不同用户数下，Sum-Rate 随 AirComp 资源占比的演进规律。
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 统一 X 轴：AirComp 资源占比 (D_a / 128)
# 分配点对应 D_a = [0, 32, 64, 96, 128]
x_ratio = np.array([0, 0.25, 0.5, 0.75, 1.0])

# 2. 录入 25dB 下的 Sum-Rate 数据
# K=2: 依次对应 D_a = 0, 32, 64, 96, 128
rate_2ue = np.array([10.66, 16.62, 17.02, 17.81, 19.36])

# K=4: 依次对应 D_a = 0, 32, 64, 96, 128
rate_4ue = np.array([13.36, 20.82, 15.65, 17.28, 25.55])

# K=8: 依次对应 D_a = 0, 32, 64, 96, 128
# 注意：最后一个点使用无 MSE 毒害的 17.09
rate_8ue = np.array([2.94, 9.19, 9.50, 10.89, 17.09])

# 3. 画纸与字体设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'font.size': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

fig, ax = plt.subplots(figsize=(8, 6))

# 4. 绘制三条核心曲线 (使用不同 Marker 和线型强化区分度)
ax.plot(x_ratio, rate_2ue, marker='s', markersize=8, linewidth=2.5, 
        color='#1f77b4', mfc='white', markeredgewidth=2, label='$K=2$ (Abundant: Monotonic)')

ax.plot(x_ratio, rate_4ue, marker='o', markersize=9, linewidth=3.5, 
        color='#d62728', mfc='white', markeredgewidth=2, label='$K=4$ (Scarce: N-Shape Phase Transition)')

ax.plot(x_ratio, rate_8ue, marker='^', markersize=9, linewidth=2.5, 
        color='#2ca02c', mfc='white', markeredgewidth=2, label='$K=8$ (Starved: Anchor Collapse)')

# 5. 图形标注与修饰
ax.set_xlabel('AirComp Resource Ratio $\gamma = D_a / D_{tot}$', fontweight='bold')
ax.set_ylabel('Sum-Rate @ 25 dB (bps/Hz)', fontweight='bold')
ax.set_xticks(x_ratio)
ax.set_xticklabels(['0\n(Pure FDMA)', '0.25', '0.50', '0.75', '1.0\n(Pure AirComp)'])

# 突出 K=4 的 Local Optimum 甜点
ax.annotate('Local Optimum\nHybrid (24, 32)', xy=(0.25, 20.82), xytext=(0.15, 23),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
            fontsize=10, fontweight='bold', color='#d62728')

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

plt.tight_layout()

# 保存
os.makedirs("results", exist_ok=True)
# plt.savefig("results/fig3_k_evolution.pdf", format='pdf', dpi=300)
plt.savefig("results/fig3_k_evolution.png", format='png', dpi=300)
print("Plot saved successfully!")