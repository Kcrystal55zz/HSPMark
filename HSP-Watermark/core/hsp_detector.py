import torch
import torch.nn.functional as F

class ContextAwareHSPDetector:
    def __init__(self, mlp_net, llm_backbone, llm_tokenizer, embeddings_weight):
        self.mlp_net = mlp_net
        self.llm_backbone = llm_backbone
        self.llm_tokenizer = llm_tokenizer
        self.embeddings = embeddings_weight.to(torch.float32)
        
        self.mlp_net.eval()
        self.llm_backbone.eval()

    def extract_message(self, text: str, message_dim: int, min_prefix_len=5):
        """
        优雅盲提取：利用生成时的自然分布假定，反向榨取多比特信息。
        :param text: 待测文本
        :param message_dim: 提取的秘钥比特长度
        :return: 提取出的比特张量 [-1, 1]
        """
        device = next(self.mlp_net.parameters()).device
        input_ids = self.llm_tokenizer.encode(text, return_tensors="pt")[0].to(device)
        total_tokens = len(input_ids)
        
        if total_tokens <= min_prefix_len: 
            return torch.ones(1, message_dim).to(device)

        extracted_scores = torch.zeros(1, message_dim).to(device)
        valid_tokens_count = 0

        with torch.no_grad():
            # 提前拿全词表对应的词向量
            text_embeds = self.embeddings[input_ids]

            for t in range(min_prefix_len, total_tokens):
                # 1. 拿到当前词对应的上下文
                prefix_ids = input_ids[:t].unsqueeze(0)
                sem = self.llm_backbone(prefix_ids).last_hidden_state[:, -1, :].to(torch.float32)
                sem = F.normalize(sem, p=2, dim=-1)

                # 2. MLP 算出当时的动态投影矩阵 P_t [hidden_dim, message_dim]
                P_t = self.mlp_net(sem).squeeze(0) 

                # 3. 拿到该位置上 真实生成的词的 Embedding [hidden_dim]
                token_embed = text_embeds[t] 

                # 4. 关键：反向映射，直接拿词向量与 P_t 相乘得到碎片信息 m_t [message_dim]
                token_msg_score = torch.matmul(token_embed.unsqueeze(0), P_t)
                
                extracted_scores += token_msg_score
                valid_tokens_count += 1

        if valid_tokens_count == 0:
            return torch.ones(1, message_dim).to(device)

        # 5. 全局平均汇总，恢复比特极性
        avg_scores = extracted_scores / valid_tokens_count
        extracted_message = torch.sign(avg_scores)
        extracted_message[extracted_message == 0] = 1.0 
        
        return extracted_message
