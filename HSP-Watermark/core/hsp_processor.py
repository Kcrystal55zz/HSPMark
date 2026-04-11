import torch
import re
from transformers import LogitsProcessor
from .utils_crypto import get_seed_from_tensor, generate_orthogonal_matrix

class SemanticOrthogonalLogitsProcessor(LogitsProcessor):
    def __init__(self, sentence_model, mlp_net, llm_tokenizer, message: torch.Tensor, alpha=2.0, top_k=50):
        self.sentence_model = sentence_model
        self.mlp_net = mlp_net.eval()
        self.tokenizer = llm_tokenizer
        self.message = message.view(1, -1).to(torch.float32)
        self.alpha = alpha
        self.top_k = top_k
        self.vocab_size = len(llm_tokenizer)
        self.message_dim = self.message.shape[1] # 获取比特维度

    def _get_last_complete_sentence(self, text: str) -> str:
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
        device = scores.device
        batch_size = input_ids.shape[0]

        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        anchor_sentence = self._get_last_complete_sentence(current_text)

        # 动态频率惩罚 (抗重复)
        current_incomplete_sentence = current_text.split(anchor_sentence)[-1]
        recent_ids = self.tokenizer.encode(current_incomplete_sentence, add_special_tokens=False)
        freq_counts = torch.zeros(self.vocab_size, device=device)
        for tid in recent_ids[-20:]: 
            if tid < self.vocab_size:
                freq_counts[tid] += 1.0

        with torch.no_grad():
            sent_embed = self.sentence_model.encode(anchor_sentence, convert_to_tensor=True)
            sent_embed = sent_embed.to(device).unsqueeze(0)

            robust_seed_tensor = self.mlp_net(sent_embed)
            seed = get_seed_from_tensor(robust_seed_tensor)

            U_i = generate_orthogonal_matrix(seed, self.message_dim, self.vocab_size, device)
            
            # 【你的优化方案】：直接除以 message_dim，防止高维度的极值爆炸
            raw_bias = torch.matmul(self.message, U_i) / self.message_dim
            
            mean = raw_bias.mean(dim=-1, keepdim=True)
            std = raw_bias.std(dim=-1, keepdim=True) + 1e-8
            normalized_bias = (raw_bias - mean) / std
            normalized_bias = normalized_bias.repeat(batch_size, 1)

            # 减去重复词的惩罚
            penalty = freq_counts.unsqueeze(0) * 1.5  
            penalized_bias = normalized_bias - penalty

            # Top-K 掩码保护 PPL
            top_k_vals, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, top_k_indices, True)
            
            final_bias = torch.where(mask, penalized_bias, torch.zeros_like(penalized_bias))

        return scores + (self.alpha * final_bias).to(scores.dtype)
