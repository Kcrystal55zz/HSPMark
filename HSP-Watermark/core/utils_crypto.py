import torch
import torch.nn as nn
import hashlib
import numpy as np

class RobustSemanticMLP(nn.Module):
    """
    用于提取稳健语义特征的 MLP（你需要用对比学习提前训练它，使其抵抗重写攻击）。
    输入：Sentence-BERT 的句向量 (例如 384 维或 768 维)
    输出：降维且稳健的语义特征向量
    """
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # L2 归一化，保证特征处于单位球面上，增强鲁棒性
        out = self.net(x)
        return torch.nn.functional.normalize(out, p=2, dim=-1)

def get_seed_from_tensor(tensor: torch.Tensor) -> int:
    """
    将稳健的语义张量映射为一个确定的随机数种子 (PRNG Seed)。
    使用 round(4) 抹除浮点数微小误差，保证重写后即使特征有极小抖动，也能映射到同一个种子。
    """
    tensor_np = tensor.detach().cpu().numpy().flatten()
    # 保留 4 位小数以容忍微小扰动
    tensor_bytes = np.round(tensor_np, decimals=4).tobytes()
    hash_hex = hashlib.md5(tensor_bytes).hexdigest()
    # 取前 8 个字符转为 32 位整数种子
    return int(hash_hex[:8], 16)

def generate_orthogonal_matrix(seed: int, message_dim: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    """
    根据给定的种子，生成一个标准正态分布的伪随机打分矩阵 (正交空间)。
    形状: [message_dim, vocab_size]
    """
    # 记录当前的随机状态，避免污染全局随机数生成器
    rng_state = torch.get_rng_state()
    
    torch.manual_seed(seed)
    # 生成均值为0，方差为1的正态分布白噪声矩阵
    U = torch.randn((message_dim, vocab_size), device=device, dtype=torch.float32)
    
    # 恢复全局随机状态
    torch.set_rng_state(rng_state)
    return U
