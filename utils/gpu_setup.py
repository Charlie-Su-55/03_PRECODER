"""
GPU Configuration for Pure PyTorch & RTX 5090
Refined for 03_PRECODER project.
"""
import torch
import os

def setup_gpu():
    # 1. 显存分配优化
    # 告诉 PyTorch 使用扩充的显存分配器，这对 5090 处理超大 Batch 的信道张量有好处
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 2. 检查设备
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        
        print(f"🚀 [03_PRECODER] GPU Setup Initialized")
        print(f"   --> Found {device_count} GPU(s)")
        print(f"   --> Principal Device: {device_name}")
        print(f"   --> CUDA Version: {torch.version.cuda}")
        
        # 3. 设置默认设备 (可选)
        device = torch.device("cuda:0")
        
        # 4. 彻底删除对 TensorFlow 的任何引用
        # 无需再写 tf.config... 
        
        return device
    else:
        print("⚠️ [Warning] No GPU detected. Running on CPU.")
        return torch.device("cpu")

if __name__ == "__main__":
    setup_gpu()