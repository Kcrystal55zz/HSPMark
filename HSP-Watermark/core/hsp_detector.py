import torch
from typing import Tuple

class HSPWatermarkDetector:
    def __init__(self, embeddings_weight: torch.Tensor, p_matrix: torch.Tensor):
        """
        全息语义投影水印提取器。
        采用全局平均池化，天生具备免同步(Synchronization-Free)和抗增删(Deletion-Resilient)特性。
        
        :param embeddings_weight: 语言模型的 Embedding 权重 [vocab_size, hidden_dim]
        :param p_matrix: 与生成时一致的私钥矩阵 [hidden_dim, message_length]
        """
        self.embeddings = embeddings_weight
        self.p_matrix = p_matrix

    def detect(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从被检测文本的 Token IDs 中提取多比特水印信息。
        
        :param input_ids: 待测文本的 Token IDs 序列 [sequence_length]
        :return: (提取的极性张量 [-1, 1], 原始反投影浮点得分)
        """
        # 确保张量在同一设备
        device = self.embeddings.device
        input_ids = input_ids.to(device)
        self.p_matrix = self.p_matrix.to(device)
        
        # 1. 查找文本中所有词的 Embedding 向量 [sequence_length, hidden_dim]
        text_embeds = self.embeddings[input_ids]
        
        # 2. 全局平均池化，获取该段文本的宏观语义向量 [hidden_dim]
        e_doc = torch.mean(text_embeds, dim=0)
        
        # 3. 反向投影，通过私钥矩阵解密得分 [message_length]
        extracted_scores = torch.matmul(e_doc, self.p_matrix)
        
        # 4. 取符号恢复极性信息
        extracted_message = torch.sign(extracted_scores)
        
        # 处理恰好为 0 的边界情况（默认归为 1）
        extracted_message[extracted_message == 0] = 1.0 
        
        return extracted_message, extracted_scores