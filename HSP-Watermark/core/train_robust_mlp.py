import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from utils_crypto import RobustSemanticMLP
import os

class STSDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                # 归一化 score 到 [-1, 1] 之间，配合 Cosine Embedding Loss
                # 原 score: 0~5。5分代表完全一样(1.0)，0分代表完全无关(-1.0)
                norm_score = (item['score'] / 2.5) - 1.0 
                self.data.append({
                    's1': item['sentence1'],
                    's2': item['sentence2'],
                    'score': norm_score
                })

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def train_mlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 初始化模型
    # 使用轻量级高精度句向量模型
    sent_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    sent_model.eval() # 冻结 BERT，只训练我们的 MLP
    
    # 句向量维度 384，映射到 128 维稳健空间
    mlp = RobustSemanticMLP(input_dim=384, hidden_dim=256, output_dim=128).to(device)
    
    # 2. 准备数据 (请将你的训练集路径填入这里)
    dataset_path = "./datasets/train.jsonl" 
    if not os.path.exists(dataset_path):
        print(f"请确保数据集存在于 {dataset_path}")
        return
        
    dataloader = DataLoader(STSDataset(dataset_path), batch_size=64, shuffle=True)
    
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3)
    criterion = nn.CosineEmbeddingLoss() # 极度适合这种相似度打分任务

    # 3. 开始训练
    epochs = 10
    mlp.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            s1_texts, s2_texts, scores = batch['s1'], batch['s2'], batch['score'].to(device).float()
            
            with torch.no_grad():
                emb1 = sent_model.encode(s1_texts, convert_to_tensor=True).to(device)
                emb2 = sent_model.encode(s2_texts, convert_to_tensor=True).to(device)
            
            # 【修复报错的关键代码】：克隆 tensor 使其脱离 inference 模式，以便允许反向传播
            emb1 = emb1.clone()
            emb2 = emb2.clone()
            
            # 经过 MLP 提取稳健特征
            out1 = mlp(emb1)
            out2 = mlp(emb2)
            
            # 计算损失：如果 scores 接近 1，out1 和 out2 的余弦相似度也要接近 1
            loss = criterion(out1, out2, scores)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # 4. 保存权重
    os.makedirs("./results", exist_ok=True)
    torch.save(mlp.state_dict(), "./results/robust_mlp.pth")
    print("MLP 预训练完成，已保存至 ./results/robust_mlp.pth")

if __name__ == "__main__":
    train_mlp()
