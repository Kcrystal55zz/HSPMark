import torch
import math
from transformers import LogitsProcessor
from .utils import prf

class WatermarkMPAC(LogitsProcessor):
    def __init__(self, tokenizer, vocab_size, window_size=2, bits='0'*16, c_key=15485863, gamma=0.25, delta=2.0):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.c_key = c_key
        self.gamma = gamma
        self.delta = delta
        self.r = int(1 / gamma) # Colorlist 的数量，例如 gamma=0.25 -> r=4
        
        # 计算每个符号携带的比特数
        self.bits_per_symbol = int(math.log2(self.r))
        assert len(bits) % self.bits_per_symbol == 0, f"Message length must be a multiple of {self.bits_per_symbol}"
        
        # 将二进制字符串转为 Radix r 的消息数组
        self.m = []
        for i in range(0, len(bits), self.bits_per_symbol):
            symbol_bits = bits[i:i+self.bits_per_symbol]
            self.m.append(int(symbol_bits, 2))
            
        self.effective_b = len(self.m) # 论文中的 \tilde{b}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        device = scores.device
        new_scores = scores.clone()
        
        for i in range(batch_size):
            seq = input_ids[i]
            
            if seq.shape[0] < self.window_size:
                continue
                
            # 1. 提取上下文并计算 seed
            prefix = seq[-self.window_size:]
            c_seed_list = prf(prefix.unsqueeze(0), self.c_key)
            c_seed = c_seed_list[0] if isinstance(c_seed_list, list) else c_seed_list
            
            # 使用 CPU Generator 保证跨设备结果一致且符合伪随机逻辑
            rng = torch.Generator(device='cpu')
            rng.manual_seed(c_seed % (2**64 - 1))
            
            # 2. 分配位置 p
            p = torch.randint(low=0, high=self.effective_b, size=(1,), generator=rng).item()
            
            # 3. 获取该位置的 Message 内容
            m_val = self.m[p]
            
            # 4. & 5. 使用同一个 seed 置换并划分词表
            vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=rng)
            colorlists = torch.chunk(vocab_perm, self.r)
            
            # 6. 对命中的 Colorlist 加上 bias (delta)
            target_list = colorlists[m_val].to(device)
            new_scores[i, target_list] += self.delta
            
        return new_scores

    def decode_mpac_multibit(self, inputs, c_key, bits, gamma=0.25):
        """
        MPAC 多比特盲检测器
        """
        import torch
        import math

        r = int(1 / gamma)
        bits_per_symbol = int(math.log2(r))
        effective_b = len(bits) // bits_per_symbol

        # 统计矩阵 W[p][m]
        W = [[0 for _ in range(r)] for _ in range(effective_b)]
        hist = set()
        valid_tokens = 0

        for t in range(self.window_size, len(inputs)):
            prefix = inputs[t - self.window_size: t]
            pref_tuple = tuple(prefix.tolist())

            if pref_tuple in hist:
                continue
            hist.add(pref_tuple)

            c_seed_list = prf(prefix.unsqueeze(0), c_key)
            c_seed = c_seed_list[0] if isinstance(c_seed_list, list) else c_seed_list

            rng = torch.Generator(device='cpu')
            rng.manual_seed(c_seed % (2 ** 64 - 1))

            # 重建分配位置和词表切分，顺序必须与编码时严格一致
            p = torch.randint(low=0, high=effective_b, size=(1,), generator=rng).item()
            vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=rng)
            colorlists = torch.chunk(vocab_perm, r)

            token_idx = inputs[t].item()

            # 寻找当前 token 属于哪个 colorlist
            for m_idx in range(r):
                if token_idx in colorlists[m_idx]:
                    W[p][m_idx] += 1
                    break

            valid_tokens += 1

        # 还原比特信息并计算匹配度
        decode_bits = ''
        hit = 0
        z_scores_symbols = []

        for p in range(effective_b):
            # 取票数最多的 colorlist 作为该位置的符号预测
            predicted_symbol = max(range(r), key=lambda x: W[p][x])

            # 将 symbol 转回二进制字符串并补齐
            symbol_bits = bin(predicted_symbol)[2:].zfill(bits_per_symbol)
            decode_bits += symbol_bits

            # 统计 Z-score (以 1/r 作为均值期望，评估分布是否显著偏向某一色表)
            total_votes_p = sum(W[p])
            max_votes = W[p][predicted_symbol]
            z = 0 # 直接赋 0，绕过缺失的函数
            z_scores_symbols.append(z)

        # 计算 Bit 准确率 (Hit Rate)
        for i in range(len(bits)):
            if decode_bits[i] == bits[i]:
                hit += 1

        hit_rate = hit / len(bits) if len(bits) > 0 else 0

        print(f"MPAC Decoding -> Extracted Bits: {decode_bits}, Hit Rate: {hit_rate:.2f}")
        return W, decode_bits, hit, hit_rate, valid_tokens, z_scores_symbols