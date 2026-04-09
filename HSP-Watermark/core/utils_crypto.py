import torch
from typing import List, Tuple

def generate_private_key(hidden_dim: int, message_length: int, device: str = 'cpu', seed: int = 42) -> torch.Tensor:
    """
    生成用于全息语义投影的正交私钥矩阵 P (正交子空间保证比特间零干扰)。
    """
    torch.manual_seed(seed)
    random_matrix = torch.randn(hidden_dim, message_length, dtype=torch.float32, device=device)
    # 使用 QR 分解获取正交矩阵分量 Q
    q_matrix, _ = torch.linalg.qr(random_matrix)
    return q_matrix

def encode_message(bit_array: List[int], device: str = 'cpu') -> torch.Tensor:
    """
    将二进制比特数组 (如 [0, 1, 1, 0]) 转化为极性张量 ([-1, 1, 1, -1])。
    """
    msg_tensor = torch.tensor(bit_array, dtype=torch.float32, device=device)
    return msg_tensor * 2.0 - 1.0

def decode_message(extracted_tensor: torch.Tensor) -> List[int]:
    """
    将提取的极性张量还原为二进制比特数组。
    """
    bits = (extracted_tensor > 0).long().tolist()
    return bits

def calculate_ber(extracted_bits: List[int], original_bits: List[int]) -> float:
    """
    计算误码率 (Bit Error Rate, BER)。
    """
    assert len(extracted_bits) == len(original_bits), "Length of extracted and original messages must match."
    errors = sum([1 for e, o in zip(extracted_bits, original_bits) if e != o])
    return errors / len(original_bits)