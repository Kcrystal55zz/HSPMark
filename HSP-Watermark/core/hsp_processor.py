import torch
from transformers import LogitsProcessor

class HSPWatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, embeddings_weight: torch.Tensor, message_tensor: torch.Tensor, p_matrix: torch.Tensor, alpha: float = 2.0):
        """
        全息语义投影水印生成器。
        通过矩阵乘法预计算全局偏置，实现 O(1) 的生成开销。
        
        :param embeddings_weight: 语言模型的 LM Head 或输出 Embedding 权重 [vocab_size, hidden_dim]
        :param message_tensor: 编码后的极性信息向量 [-1, 1] 格式 [message_length]
        :param p_matrix: 投影私钥矩阵 [hidden_dim, message_length]
        :param alpha: 水印偏置强度系数
        """
        self.alpha = alpha
        
        # 1. 计算当前信息在高维空间中的全局投影方向向量 v_dir [hidden_dim]
        v_dir = torch.matmul(p_matrix, message_tensor)
        
        # 2. 计算词表中所有词向量在该方向上的原始投影得分 [vocab_size]
        raw_scores = torch.matmul(embeddings_weight, v_dir)
        
        # 3. 均值方差归一化，确保偏置不会过度破坏特定 Token 的自然分布
        mean_score = raw_scores.mean()
        std_score = raw_scores.std() + 1e-8
        normalized_scores = (raw_scores - mean_score) / std_score
        
        # 4. 预计算静态偏置 [vocab_size]
        self.static_bias = self.alpha * normalized_scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        将静态语义偏置应用于当前步的预测 Logits。
        """
        # 确保 device 一致
        if self.static_bias.device != scores.device:
            self.static_bias = self.static_bias.to(scores.device)
            
        # scores shape: [batch_size, vocab_size]
        # 直接利用广播机制完成批量偏置叠加
        watermarked_scores = scores + self.static_bias.unsqueeze(0)
        return watermarked_scores