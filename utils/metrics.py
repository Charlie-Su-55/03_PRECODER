import torch
import torch.nn as nn

def compute_nmse_db(H_true, H_hat):
    """
    计算 NMSE (dB)。
    注意：在验证阶段使用，记录时再转为 .item()
    """
    # 增加安全性：防止 H_true 全为 0
    error_pwr = torch.sum(torch.abs(H_true - H_hat)**2, dim=(-2, -1))
    ref_pwr = torch.sum(torch.abs(H_true)**2, dim=(-2, -1))
    nmse = error_pwr / (ref_pwr + 1e-12)
    nmse_db = 10 * torch.log10(nmse + 1e-12)
    return nmse_db.mean()

def compute_avg_gcs(H_true, H_hat):
    """
    广义余弦相似度 (Generalized Cosine Similarity)
    """
    # 自动获取维度，增强兼容性
    h1 = H_true.flatten(start_dim=2) 
    h2 = H_hat.flatten(start_dim=2)
    
    numerator = torch.abs(torch.sum(torch.conj(h1) * h2, dim=-1))
    # 使用 linalg.vector_norm 性能更佳
    denominator = torch.linalg.vector_norm(h1, dim=-1) * torch.linalg.vector_norm(h2, dim=-1)
    rho = numerator / (denominator + 1e-12)
    return rho.mean()

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, gcs_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.gcs_weight = gcs_weight
    
    def forward(self, H_true, H_hat):
        # 1. MSE Loss
        mse = torch.mean(torch.abs(H_true - H_hat)**2)
        
        # 2. GCS Loss
        gcs_val = compute_avg_gcs(H_true, H_hat)
        gcs_loss = 1.0 - gcs_val
        
        total_loss = self.mse_weight * mse + self.gcs_weight * gcs_loss
        
        # 返回字典不带 .item()，由训练器在 log 阶段处理
        metrics = {
            'mse': mse,
            'gcs': gcs_val
        }
        
        return total_loss, metrics

# --- 🚀 关键补充：和速率计算 (Sum Rate) ---
def compute_sum_rate(H, W, noise_var):
    """
    针对 Perfect CSI 的和速率评估
    Args:
        H: [B, K, Nr, Nt] 信道矩阵 (假设 Nr=1)
        W: [B, K, Nt, 1] 预编码矩阵
        noise_var: 噪声功率标量
    """
    # 假设 Nr=1, 降维到 [B, K, Nt]
    H = H.squeeze(2) 
    W = W.squeeze(-1)
    
    # 计算有用信号功率 |h_k * w_k|^2
    # signal: [B, K]
    signal_pwr = torch.abs(torch.sum(H * W, dim=-1))**2
    
    # 计算干扰功率 (Interference)
    # 简化版：这里需要根据你的多用户逻辑计算 sum_{j != k} |h_k * w_j|^2
    # 我们先预留这个接口，等搬运 Precoding 逻辑时细化
    
    sinr = signal_pwr / noise_var # 假设无干扰或零迫近
    rate = torch.log2(1 + sinr)
    return rate.sum(dim=-1).mean()