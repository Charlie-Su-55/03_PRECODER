# configs/exp_4ue_128sc.py
"""
Experiment Config: 4 UE × 128 Subcarriers
Architecture: Deep Unfolded WMMSE + Hybrid FDMA-AirComp Feedback
"""

class HybridPrecoderConfig:
    # ========================================================
    # 1. 物理层参数
    # ========================================================
    K_USERS = 4
    ANTENNAS = 32
    SUBCARRIERS = 128 
    CARRIER_FREQ = 3.5e9
    SPEED = 1.0

    # ========================================================
    # 2. 频谱资源分配
    # ----------------------------------------------------------
    # 当前: 纯FDMA 对照 (DIM_AIRCOMP_SHARED = 0)
    # 切换为 Hybrid: 满足 K_USERS*FDMA + AC = SUBCARRIERS
    # 切换为 纯AirComp: FDMA=0, AC=512
    # ========================================================
    DIM_FDMA_PER_USER = 32 # 0 # 24 # 16 # 32 # 0
    DIM_AIRCOMP_SHARED = 0 # 128 # 32 # 64 # 0 # 128
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
    BATCH_SIZE = 48 # 32             # 8UE/512SC下5090的安全值, 防OOM
    LR = 1e-4
    TOTAL_STEPS = 20000
    VAL_INTERVAL = 500

    # 优化器辅助超参
    WARMUP_PCT = 0.35           # OneCycleLR预热比例
    GRAD_CLIP = 1.0             # 梯度裁剪阈值

    # ========================================================
    # 5. SNR 课程学习
    # ========================================================
    STAGE1_STEPS = 12000
    STAGE1_SNR = (20.0, 25.0)
    STAGE2_STEPS = 16000
    STAGE2_SNR = (10.0, 25.0)
    STAGE3_SNR = (0.0, 25.0)
    VAL_SNR_LIST = [0, 10, 20, 25]

    # ========================================================
    # 6. 文件路径
    # ========================================================
    # EXP_NAME = (
    #     f"uwmmse_{K_USERS}ue_{SUBCARRIERS}sc"
    #     f"_fdma{DIM_FDMA_PER_USER}_ac{DIM_AIRCOMP_SHARED}"
    #     f"_L{NUM_UNFOLD_LAYERS}"
    # )
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

    # ========================================================
    # 7. Swin Baseline 专用超参数
    # ========================================================
    # Angular-Delay 域截断维度 (压缩 Nsc=128 → SWIN_NC=32 个主延时径)
    # 因为信道的 delay spread 通常 << OFDM symbol period, 后面延时径基本是 0
    SWIN_NC = 32
    
    # Patch embedding 配置
    SWIN_PATCH_SIZE = 4  # 4×4 patch
    
    # Transformer 配置 (单层 stage, 简化版 Swin)
    SWIN_EMBED_DIMS = [128]      # 单 stage 嵌入维度
    SWIN_NUM_HEADS = [4]         # 注意力头数
    SWIN_NUM_BLOCKS = [4]        # Swin block 数量
    SWIN_WINDOW_SIZE = 4         # 窗口 attention 大小
    SWIN_MLP_RATIO = 4.0         # MLP 扩展比例
    
    # Codeword 维度 = 每个 UE 反馈的实数维度
    # 对齐 Pure FDMA 设定: 每 UE 反馈 D_f=32 复数 = 64 实数
    # 注意: 这里要跟你的 DIM_FDMA_PER_USER 对齐才公平!
    SWIN_CODEWORD_DIM = DIM_FDMA_PER_USER * 2  # 32 复数 = 64 实数

    # ========================================================
    # 8. CsiNet+ Baseline 专用超参数
    # ========================================================
    # Angular-Delay 域截断维度 (跟 Swin 对齐)
    CSINET_NC = 32
    
    # Encoder/Decoder 通道数 (CsiNet+ 论文标准配置)
    CSINET_ENC_CHANNELS = [16, 8, 4, 2]
    CSINET_DEC_CHANNELS = [2, 4, 8, 16]
    
    # 注意: CsiNet+ 的 codeword 维度自动从 DIM_FDMA_PER_USER 计算
    # 不需要单独定义, 在 model __init__ 里自动 = DIM_FDMA_PER_USER * 2