import torch
import torch.nn.functional as F
import re
from transformers import LogitsProcessor
import nltk

class SemanticOrthogonalLogitsProcessor(LogitsProcessor):
    def __init__(self, sentence_model, mlp_net, llm_tokenizer, message: torch.Tensor, alpha=2.0, top_k=50, window_size=2):
        self.sentence_model = sentence_model
        self.mlp_net = mlp_net.eval()
        self.tokenizer = llm_tokenizer
        self.message = message.view(1, -1).to(torch.float32)
        self.alpha = alpha
        self.top_k = top_k
        self.vocab_size = len(llm_tokenizer)
        self.message_dim = self.message.shape[1] 
        self.window_size = window_size # 滚动窗口大小
        
        self.device = next(self.mlp_net.parameters()).device
        self.message = self.message.to(self.device)
        self.master_key = 15485863 
        self.proj_dim = 2048        
        
        with torch.no_grad():
            dummy_text = "dummy"
            dummy_embed = self.sentence_model.encode(dummy_text, convert_to_tensor=True).unsqueeze(0).to(self.device)
            mlp_out_dim = self.mlp_net(dummy_embed).shape[-1]
            
        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.master_key)
        self.W = torch.randn(self.message_dim, mlp_out_dim, self.proj_dim, generator=rng, device=self.device)
        self.mapping = torch.randint(0, self.proj_dim, (self.vocab_size,), generator=rng, device=self.device)

    def _get_robust_context(self, text: str) -> str:
        """使用滚动窗口获取鲁棒的上下文 Anchor"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return text.strip()
        
        # 排除最后一句不完整的句子，取前面的 window_size 句话
        complete_sentences = sentences[:-1] 
        # 截取最近的 window_size 句拼接
        context_sentences = complete_sentences[-self.window_size:]
        return " ".join(context_sentences)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        anchor_text = self._get_robust_context(current_text)

        # 提取当前正在生成的未完成部分
        current_incomplete_sentence = current_text[len(anchor_text):].strip() if anchor_text in current_text else current_text
        recent_ids = self.tokenizer.encode(current_incomplete_sentence, add_special_tokens=False)
        freq_counts = torch.zeros(self.vocab_size, device=self.device)
        for tid in recent_ids[-20:]: 
            if tid < self.vocab_size:
                freq_counts[tid] += 1.0

        with torch.no_grad():
            sent_embed = self.sentence_model.encode(anchor_text, convert_to_tensor=True)
            sent_embed = sent_embed.to(self.device).unsqueeze(0)

            robust_tensor = self.mlp_net(sent_embed)
            V = F.normalize(robust_tensor, p=2, dim=-1) 
            
            H = torch.einsum('bd, kdp -> bkp', V, self.W)
            U_i = H[:, :, self.mapping]
            
            raw_bias = torch.sum(self.message.unsqueeze(-1) * U_i, dim=1) / self.message_dim
            
            mean = raw_bias.mean(dim=-1, keepdim=True)
            std = raw_bias.std(dim=-1, keepdim=True) + 1e-8
            normalized_bias = (raw_bias - mean) / std
            normalized_bias = normalized_bias.repeat(batch_size, 1)

            penalty = freq_counts.unsqueeze(0) * 1.5  
            penalized_bias = normalized_bias - penalty

            # --- 改进: 熵自适应与截断 ---
            probs = F.softmax(scores, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
            # 设定熵缩放系数，确信(熵低)时偏置减弱，迷茫(熵高)时偏置增强
            entropy_scale = torch.clamp(entropy / 2.0, min=0.2, max=1.2)
            
            top_k_vals, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, top_k_indices, True)
            
            # 计算最终偏置并执行截断，防止突变拉高 PPL
            scaled_bias = penalized_bias * entropy_scale
            scaled_bias = torch.clamp(scaled_bias, min=-2.5, max=2.5) 
            
            final_bias = torch.where(mask, scaled_bias, torch.zeros_like(scaled_bias))

        return scores + (self.alpha * final_bias).to(scores.dtype)
