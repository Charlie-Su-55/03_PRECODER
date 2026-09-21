# configs/base_config.py

class BaseConfig:
    # --- 1. 物理层参数 ---
    K_USERS = 4
    ANTENNAS = 32
    SUBCARRIERS = 256
    CARRIER_FREQ = 3.5e9
    SPEED = 1.0 
    
    # --- 2. 频谱资源分配 (带宽开销对齐) ---
    DIM_FDMA_PER_USER = 32 
    DIM_AIRCOMP_SHARED = 128
    # Swin 对照组专属开销: 128实数 = 64复数符号/用户. 4 * 64 = 256 满载
    SWIN_CODEWORD_DIM = 128 
    
    # --- 3. 模型超参数 ---
    D_MODEL = 768            
    NUM_HEADS = 12           
    NUM_ENCODER_LAYERS = 6    
    NUM_DECODER_LAYERS = 4    
    NUM_ITERATIONS = 5        
    DROPOUT = 0.1
    
    # Swin 专属超参数
    SWIN_NC = 32           # 截断的延时域维度
    SWIN_PATCH_SIZE = 2
    SWIN_EMBED_DIMS = [64, 128]
    SWIN_NUM_BLOCKS = [2, 4]
    SWIN_NUM_HEADS = [4, 8]
    SWIN_WINDOW_SIZE = 4
    SWIN_MLP_RATIO = 4
    
    # --- 4. 训练总控 ---
    TOTAL_POWER = 1.0        
    BATCH_SIZE = 64          
    LR = 5e-5                
    TOTAL_STEPS = 100000     
    VAL_INTERVAL = 500       

    # --- 5. 阶梯式课程学习策略 ---
    STAGE1_STEPS = 60000
    STAGE1_SNR = (20.0, 25.0)
    
    STAGE2_STEPS = 80000
    STAGE2_SNR = (10.0, 25.0)
    
    STAGE3_SNR = (0.0, 25.0)
    VAL_SNR_LIST = [0, 10, 20, 25] 

    # --- 6. 文件路径 ---
    SAVE_DIR = "ckp"