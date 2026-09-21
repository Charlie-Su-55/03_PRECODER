# utils/__init__.py

# 1. ⚠️ 第一优先级：初始化 GPU 配置 (RTX 5090 显存优化)
from .gpu_setup import setup_gpu
setup_gpu() 

# 2. 导出核心组件
from .channel_generator import CDLChannelGenerator_Feedback
from .layers import DFTLayer, IDFTLayer
from .metrics import compute_nmse_db, compute_avg_gcs, CombinedLoss, compute_sum_rate

# 3. 定义外部可见接口
__all__ = [
    'CDLChannelGenerator_Feedback',
    'DFTLayer', 
    'IDFTLayer',
    'compute_nmse_db',
    'compute_avg_gcs',
    'compute_sum_rate', # 别忘了把这个刚加的“和速率”指标也暴露出来
    'CombinedLoss'
]