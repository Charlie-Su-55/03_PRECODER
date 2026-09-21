# configs/exp_2ue_128sc.py
"""
Experiment Config: 2 UE × 128 Subcarriers
Architecture: Deep Unfolded WMMSE + Hybrid FDMA-AirComp Feedback
"""

class HybridPrecoderConfig:
    # ========================================================
    # 1. 物理层参数
    # ========================================================
    K_USERS = 2
    ANTENNAS = 32
    SUBCARRIERS = 128 
    CARRIER_FREQ = 3.5e9
    SPEED = 1.0

    # ========================================================
    # 2. 频谱资源分配
    # ----------------------------------------------------------
    # 当前: 纯FDMA 对照 (DIM_AIRCOMP_SHARED = 0)
    # 切换为 Hybrid: FDMA=32, AC=256, 满足 K_USERS*FDMA + AC = 512
    # 切换为 纯AirComp: FDMA=0, AC=512
    # ========================================================
    DIM_FDMA_PER_USER = 64 # 48 # 32 # 16 # 0 
    DIM_AIRCOMP_SHARED = 0 # 32 # 64 # 96 # 128 
    NUM_SUBBANDS = 4 

    # ========================================================
    # 3. 模型超参数
    # ========================================================
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4      # 历史保留, Unfolded版本未使用
    NUM_ITERATIONS = 5          # 历史保留, Unfolded版本未使用
    DROPOUT = 0.1

    # Unfolded WMMSE 核心超参
    NUM_UNFOLD_LAYERS = 5       # WMMSE展开层数, 推荐范围 3~8
    GNN_AGG_DIM = 48            # GNN聚合模块的协方差特征维度 d_a

    # ========================================================
    # 4. 训练总控
    # ========================================================
    TOTAL_POWER = 1.0
    BATCH_SIZE = 32             # 8UE/512SC下5090的安全值, 防OOM
    LR = 5e-5 
    TOTAL_STEPS = 40000
    VAL_INTERVAL = 1000

    # 优化器辅助超参
    WARMUP_PCT = 0.1 # 0.15 # 0.2 # 0.25 # 0.35         # OneCycleLR预热比例
    GRAD_CLIP = 0.5 # 0.4 # 0.3 # 0.2 # 0.1           # 梯度裁剪阈值

    # ========================================================
    # 5. SNR 课程学习
    # ========================================================
    STAGE1_STEPS = 30000
    STAGE1_SNR = (20.0, 25.0)
    STAGE2_STEPS = 38000
    STAGE2_SNR = (10.0, 25.0)
    STAGE3_SNR = (0.0, 25.0)
    VAL_SNR_LIST = [0, 10, 20, 25]

    # ========================================================
    # 6. 文件路径
    # ========================================================
    EXP_NAME = (
        f"uwmmse_{K_USERS}ue_{SUBCARRIERS}sc"
        f"_fdma{DIM_FDMA_PER_USER}_ac{DIM_AIRCOMP_SHARED}"
        f"_L{NUM_UNFOLD_LAYERS}"
        f"_sb{NUM_SUBBANDS}" 
    )
    SAVE_DIR = f"ckp/{EXP_NAME}"
    SAVE_PATH = f"{SAVE_DIR}/uwmmse_latest.pth"
    BEST_PATH = f"{SAVE_DIR}/uwmmse_best.pth"
    LOG_DIR = f"logs/{EXP_NAME}"