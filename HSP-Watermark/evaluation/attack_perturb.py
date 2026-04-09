import random
from typing import List

class PerturbationAttacker:
    def __init__(self, seed: int = None):
        """
        词级别扰动攻击器，用于验证水印算法在位置错乱/长度改变时的鲁棒性。
        """
        if seed is not None:
            random.seed(seed)

    def attack_drop(self, tokens: List[int], drop_ratio: float = 0.1) -> List[int]:
        """
        随机删除攻击 (Token Deletion)
        用于测试算法应对直接截断和信息丢失的能力。
        """
        if drop_ratio <= 0.0:
            return tokens
        
        attacked_tokens = [t for t in tokens if random.random() > drop_ratio]
        
        # 防止将序列完全删空
        if len(attacked_tokens) == 0 and len(tokens) > 0:
            attacked_tokens = [tokens[0]]
            
        return attacked_tokens

    def attack_swap(self, tokens: List[int], swap_ratio: float = 0.1) -> List[int]:
        """
        随机交换攻击 (Token Swapping)
        用于破坏 N-gram 结构，测试对局部语序颠倒的抗性。
        """
        if swap_ratio <= 0.0 or len(tokens) < 2:
            return tokens
            
        attacked_tokens = tokens.copy()
        n = len(attacked_tokens)
        num_swaps = int(n * swap_ratio)
        
        for _ in range(num_swaps):
            idx = random.randint(0, n - 2)
            attacked_tokens[idx], attacked_tokens[idx+1] = attacked_tokens[idx+1], attacked_tokens[idx]
            
        return attacked_tokens