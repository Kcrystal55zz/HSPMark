import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

class ContextAwareHSPLogitsProcessor(LogitsProcessor):
    def __init__(self, mlp_net, llm_backbone, message, embeddings_weight, alpha=2.0):
        self.mlp_net = mlp_net
        self.llm_backbone = llm_backbone 
        # [1, message_dim, 1] 预留最后一维用于矩阵乘法
        self.message = message.view(1, -1, 1).to(torch.float32) 
        self.embeddings = embeddings_weight.to(torch.float32)
        self.alpha = alpha
        
        self.mlp_net.eval()
        self.llm_backbone.eval()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            # 1. 提取大模型本身的隐藏层语义特征 C_t
            outputs = self.llm_backbone(input_ids)
            semantic_embeds = outputs.last_hidden_state[:, -1, :].to(torch.float32)
            semantic_embeds = F.normalize(semantic_embeds, p=2, dim=-1)

            # 2. MLP 输出动态投影矩阵 P_t [batch_size, hidden_dim, message_dim]
            P_t = self.mlp_net(semantic_embeds)

            # 3. 投影矩阵乘上机密比特串，得到当前的偏置方向 V_t
            msg_batch = self.message.repeat(batch_size, 1, 1).to(device)
            v_dir_t = torch.bmm(P_t, msg_batch).squeeze(-1) # [batch_size, hidden_dim]

            # 4. 计算词表的投影分布并进行标准归一化
            raw_scores = torch.matmul(v_dir_t, self.embeddings.T) 
            mean = raw_scores.mean(dim=-1, keepdim=True)
            std = raw_scores.std(dim=-1, keepdim=True) + 1e-8
            normalized_scores = (raw_scores - mean) / std

            bias = self.alpha * normalized_scores

        # 加回到原大模型的 Logits 上（转换回原类型防止报错）
        return scores + bias.to(scores.dtype)
