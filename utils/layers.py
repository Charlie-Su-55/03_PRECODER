import torch
import torch.nn as nn
import math

class DFTLayer(nn.Module):
    """
    兼容版 DFT 层:
    1. 默认行为: 直接矩阵乘 (和旧版一致)。
    2. 增强行为: 若指定 dim, 可在任意维度进行 DFT 变换。
    """
    def __init__(self, n_point, dim=None):
        super().__init__()
        self.n = n_point
        self.dim = dim
        
        # 预计算 DFT 矩阵 (使用 torch.pi 彻底去毒)
        n_idx = torch.arange(n_point)
        k_idx = n_idx.view(-1, 1)
        # W_N = exp(-j2π/N)
        M = torch.exp(-2j * torch.pi * n_idx * k_idx / n_point) / math.sqrt(n_point)
        self.register_buffer('F', M)

    def forward(self, x):
        # 场景 1: 保持旧版兼容性 (直接矩阵乘)
        if self.dim is None:
            return torch.matmul(self.F, x)
        
        # 场景 2: 针对特定维度做变换 (比如 [B, K, Nt, Nsc] 中的 Nt)
        # 将目标维度换到倒数第二位，与 F 矩阵对齐
        x = x.transpose(self.dim, -2)
        out = torch.matmul(self.F, x)
        return out.transpose(self.dim, -2)


class IDFTLayer(nn.Module):
    """
    兼容版 IDFT 层:
    逻辑与 DFT 对称，用于从角域/频域还原回空间域/时域。
    """
    def __init__(self, n_point, dim=None):
        super().__init__()
        self.n = n_point
        self.dim = dim
        
        n_idx = torch.arange(n_point)
        k_idx = n_idx.view(-1, 1)
        # W_N_inv = exp(j2π/N)
        M = torch.exp(2j * torch.pi * n_idx * k_idx / n_point) / math.sqrt(n_point)
        self.register_buffer('F_H', M)

    def forward(self, x):
        # 场景 1: 保持旧版兼容性
        if self.dim is None:
            return torch.matmul(self.F_H, x)
        
        # 场景 2: 针对特定维度做变换
        x = x.transpose(self.dim, -2)
        out = torch.matmul(self.F_H, x)
        return out.transpose(self.dim, -2)