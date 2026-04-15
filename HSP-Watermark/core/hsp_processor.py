import torch
import torch.nn.functional as F
import re
from transformers import LogitsProcessor

class SemanticOrthogonalLogitsProcessor(LogitsProcessor):
    def __init__(self, sentence_model, mlp_net, llm_tokenizer, message: torch.Tensor, alpha=2.0, top_k=50):
        self.sentence_model = sentence_model
        self.mlp_net = mlp_net.eval()
        self.tokenizer = llm_tokenizer
        self.message = message.view(1, -1).to(torch.float32)
        self.alpha = alpha
        self.top_k = top_k
        self.vocab_size = len(llm_tokenizer)
        self.message_dim = self.message.shape[1] 
        
        # --- 核心改造：连续投影初始化 ---
        self.device = next(self.mlp_net.parameters()).device
        self.message = self.message.to(self.device)
        self.master_key = 15485863  # 充当全局主密钥
        self.proj_dim = 2048        # 投影维度，兼顾计算效率与随机正交性
        
        # 探测 MLP 输出维度
        with torch.no_grad():
            dummy_text = "dummy"
            dummy_embed = self.sentence_model.encode(dummy_text, convert_to_tensor=True).unsqueeze(0).to(self.device)
            mlp_out_dim = self.mlp_net(dummy_embed).shape[-1]
            
        # 生成全局固定的连续投影矩阵 W 和 词表映射 (替代原来的随机正交矩阵)
        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.master_key)
        # 形状: (比特维度, MLP输出维度, 投影维度)
        self.W = torch.randn(self.message_dim, mlp_out_dim, self.proj_dim, generator=rng, device=self.device)
        # 形状: (词表大小,) -> 将每个 Token 随机映射到 2048 维的某一个槽位中
        self.mapping = torch.randint(0, self.proj_dim, (self.vocab_size,), generator=rng, device=self.device)

    def _get_last_complete_sentence(self, text: str) -> str:
        # 恢复单句切分，因为你的 MLP 是基于 STS（单句）训练的
        sentences = re.split(r'([.!?\n]+)', text)
        combined = []
        for i in range(0, len(sentences)-1, 2):
            s = sentences[i] + sentences[i+1]
            if len(s.strip()) > 5:
                combined.append(s.strip())
        if len(sentences) % 2 != 0 and len(sentences[-1].strip()) > 5:
            combined.append(sentences[-1].strip())
            
        if len(combined) >= 2:
            return combined[-2]
        elif len(combined) == 1:
            return combined[0]
        return text.strip()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        anchor_sentence = self._get_last_complete_sentence(current_text)

        current_incomplete_sentence = current_text.split(anchor_sentence)[-1] if anchor_sentence in current_text else current_text
        recent_ids = self.tokenizer.encode(current_incomplete_sentence, add_special_tokens=False)
        freq_counts = torch.zeros(self.vocab_size, device=self.device)
        for tid in recent_ids[-20:]: 
            if tid < self.vocab_size:
                freq_counts[tid] += 1.0

        with torch.no_grad():
            sent_embed = self.sentence_model.encode(anchor_sentence, convert_to_tensor=True)
            sent_embed = sent_embed.to(self.device).unsqueeze(0)

            # 1. 获取 MLP 连续特征
            robust_tensor = self.mlp_net(sent_embed)
            # 2. L2 归一化，确保特征在一个超球面上，限制异常漂移
            V = F.normalize(robust_tensor, p=2, dim=-1) 
            
            # 3. 连续投影计算 U_i (替代原本引发雪崩的离散 Seed)
            # einsum 解释: V是(1, mlp_dim), W是(K, mlp_dim, proj_dim) => H是(1, K, proj_dim)
            H = torch.einsum('bd, kdp -> bkp', V, self.W)
            
            # 4. 扩展到全词表: U_i 形状 -> (1, message_dim, vocab_size)
            U_i = H[:, :, self.mapping]
            
            # 5. 计算最终 Bias (K维向量点乘 U_i)
            # raw_bias 形状 -> (1, vocab_size)
            raw_bias = torch.sum(self.message.unsqueeze(-1) * U_i, dim=1) / self.message_dim
            
            # 标准化与惩罚逻辑保持不变
            mean = raw_bias.mean(dim=-1, keepdim=True)
            std = raw_bias.std(dim=-1, keepdim=True) + 1e-8
            normalized_bias = (raw_bias - mean) / std
            normalized_bias = normalized_bias.repeat(batch_size, 1)

            penalty = freq_counts.unsqueeze(0) * 1.5  
            penalized_bias = normalized_bias - penalty

            top_k_vals, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, top_k_indices, True)
            
            final_bias = torch.where(mask, penalized_bias, torch.zeros_like(penalized_bias))

        return scores + (self.alpha * final_bias).to(scores.dtype)
