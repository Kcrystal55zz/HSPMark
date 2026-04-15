import torch
import torch.nn.functional as F
import re

class BlindSemanticDetector:
    def __init__(self, sentence_model, mlp_net, llm_tokenizer, message_dim: int):
        self.sentence_model = sentence_model
        self.mlp_net = mlp_net.eval()
        self.tokenizer = llm_tokenizer
        self.message_dim = message_dim
        self.vocab_size = len(llm_tokenizer)
        
        # --- 与生成端保持绝对一致的初始化 ---
        self.device = next(self.mlp_net.parameters()).device
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

    def extract_message(self, text: str) -> torch.Tensor:
        sentences = re.split(r'([.!?\n]+)', text)
        combined_sentences = []
        for i in range(0, len(sentences)-1, 2):
            s = sentences[i] + sentences[i+1]
            if len(s.strip()) > 5:
                combined_sentences.append(s.strip())
        if len(sentences) % 2 != 0 and len(sentences[-1].strip()) > 5:
            combined_sentences.append(sentences[-1].strip())

        if len(combined_sentences) < 2:
            return torch.ones(1, self.message_dim).to(self.device)

        score_accumulator = torch.zeros(1, self.message_dim).to(self.device)
        valid_tokens_count = 0

        with torch.no_grad():
            for i in range(1, len(combined_sentences)):
                anchor_sentence = combined_sentences[i-1]
                target_sentence = combined_sentences[i]

                # 1. 提取连续锚点向量
                sent_embed = self.sentence_model.encode(anchor_sentence, convert_to_tensor=True)
                sent_embed = sent_embed.to(self.device).unsqueeze(0)
                robust_tensor = self.mlp_net(sent_embed)
                V = F.normalize(robust_tensor, p=2, dim=-1)

                # 2. 连续投影重建解密矩阵 U_i (软解码)
                H = torch.einsum('bd, kdp -> bkp', V, self.W)
                U_i = H[:, :, self.mapping]  # 形状: (1, K, vocab_size)

                prefix_target = " " + target_sentence if not target_sentence.startswith(" ") else target_sentence
                target_ids = self.tokenizer.encode(prefix_target, add_special_tokens=False)

                for token_id in target_ids:
                    if token_id < self.vocab_size:
                        # 累加每个 Token 对应的 K 维解密向量
                        token_vector = U_i[0, :, token_id].unsqueeze(0)
                        score_accumulator += token_vector
                        valid_tokens_count += 1

        if valid_tokens_count == 0:
            return torch.ones(1, self.message_dim).to(self.device)

        # 多比特投票提取
        extracted_message = torch.sign(score_accumulator)
        extracted_message[extracted_message == 0] = 1.0 
        return extracted_message
