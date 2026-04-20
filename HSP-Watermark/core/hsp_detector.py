import torch
import torch.nn.functional as F
import nltk

class BlindSemanticDetector:
    def __init__(self, sentence_model, mlp_net, llm_tokenizer, message_dim: int, window_size=2):
        self.sentence_model = sentence_model
        self.mlp_net = mlp_net.eval()
        self.tokenizer = llm_tokenizer
        self.message_dim = message_dim
        self.vocab_size = len(llm_tokenizer)
        self.window_size = window_size
        
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

    def extract_message(self, text: str, return_soft_score=False):
        """
        提取水印信息
        :param return_soft_score: 若为True，返回连续的置信度幅度向量；否则返回二值化比特
        """
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) < 2:
            empty_res = torch.ones(1, self.message_dim).to(self.device)
            return (empty_res, empty_res) if return_soft_score else empty_res

        score_accumulator = torch.zeros(1, self.message_dim).to(self.device)
        valid_tokens_count = 0

        with torch.no_grad():
            # 使用滚动窗口遍历段落
            for i in range(1, len(sentences)):
                # 提取前 window_size 句作为上下文 Anchor
                start_idx = max(0, i - self.window_size)
                anchor_text = " ".join(sentences[start_idx:i])
                target_sentence = sentences[i]

                sent_embed = self.sentence_model.encode(anchor_text, convert_to_tensor=True)
                sent_embed = sent_embed.to(self.device).unsqueeze(0)
                robust_tensor = self.mlp_net(sent_embed)
                V = F.normalize(robust_tensor, p=2, dim=-1)

                H = torch.einsum('bd, kdp -> bkp', V, self.W)
                U_i = H[:, :, self.mapping]  # (1, K, vocab_size)

                prefix_target = " " + target_sentence if not target_sentence.startswith(" ") else target_sentence
                target_ids = self.tokenizer.encode(prefix_target, add_special_tokens=False)

                for token_id in target_ids:
                    if token_id < self.vocab_size:
                        token_vector = U_i[0, :, token_id].unsqueeze(0)
                        score_accumulator += token_vector
                        valid_tokens_count += 1

        if valid_tokens_count == 0:
            empty_res = torch.ones(1, self.message_dim).to(self.device)
            return (empty_res, empty_res) if return_soft_score else empty_res

        # 计算软置信度均值
        soft_scores = score_accumulator / valid_tokens_count
        
        # 多比特投票提取
        extracted_message = torch.sign(soft_scores)
        extracted_message[extracted_message == 0] = 1.0 
        
        if return_soft_score:
            return extracted_message, soft_scores
        return extracted_message
