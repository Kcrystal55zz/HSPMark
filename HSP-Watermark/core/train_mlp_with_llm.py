import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

class ContextAwareWatermarkNet(nn.Module):
    def __init__(self, semantic_dim, message_dim, hidden_dim):
        super().__init__()
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(semantic_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, hidden_dim * message_dim)
        )
        
    def forward(self, semantic_embeds):
        out = self.mlp(semantic_embeds)
        return out.view(-1, self.hidden_dim, self.message_dim)

class WatermarkLoss(nn.Module):
    def __init__(self, lambda_sim=1.0, lambda_norm=1.0):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_norm = lambda_norm

    def forward(self, logits_A, logits_B, target_sim):
        logits_sim = F.cosine_similarity(logits_A, logits_B, dim=-1)
        loss_sim = F.mse_loss(logits_sim, target_sim)

        mean_A, mean_B = logits_A.mean(dim=-1), logits_B.mean(dim=-1)
        loss_mean = (mean_A ** 2).mean() + (mean_B ** 2).mean()
        
        abs_A, abs_B = logits_A.abs().mean(dim=-1), logits_B.abs().mean(dim=-1)
        loss_abs = ((abs_A - 1.0) ** 2).mean() + ((abs_B - 1.0) ** 2).mean()
        
        loss_norm = loss_mean + loss_abs
        return self.lambda_sim * loss_sim + self.lambda_norm * loss_norm, loss_sim, loss_norm

class STSDataset(Dataset):
    def __init__(self, json_path):
        self.data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line.strip())
                if 'sentence1' in item and 'sentence2' in item and 'score' in item:
                    self.data.append({
                        'sentence1': item['sentence1'],
                        'sentence2': item['sentence2'],
                        'score': float(item['score']) / 5.0 
                    })
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_PATH = "/root/autodl-tmp/HSP-Watermark/datasets/train.jsonl"  
    LLM_MODEL_NAME = "/root/autodl-tmp/huggingface_models/Llama-2-7b-hf" 
    
    # 我们需要为 run_experiments 评测准备的比特容量：16 和 24
    MESSAGE_DIMS = [16, 24] 
    BATCH_SIZE = 16
    EPOCHS = 100

    print(f"[*] 正在加载大模型底座: {LLM_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 增加 device_map 和 float16 避免加载 7B 模型时显存爆炸
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    llm_model.eval()

    embeddings_weight = llm_model.get_output_embeddings().weight.detach()
    embeddings_weight = F.normalize(embeddings_weight, p=2, dim=-1).to(torch.float32)
    HIDDEN_DIM = embeddings_weight.shape[1]
    SEMANTIC_DIM = HIDDEN_DIM  
    
    # 保证数据集正常读取
    try:
        dataset = STSDataset(DATASET_PATH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except FileNotFoundError:
        print(f"[!] 请准备数据集 {DATASET_PATH} 以进行训练。")
        return

    # 循环训练不同比特容量的模
    for msg_dim in MESSAGE_DIMS:
        print(f"\n" + "="*50)
        print(f"[*] 开始训练 {msg_dim}-bit 容量的水印密码机...")
        print("="*50)
        
        mlp_net = ContextAwareWatermarkNet(SEMANTIC_DIM, msg_dim, HIDDEN_DIM).to(device)
        optimizer = optim.Adam(mlp_net.parameters(), lr=1e-4)
        criterion = WatermarkLoss(lambda_sim=1.0, lambda_norm=0.5)

        for epoch in range(EPOCHS):
            total_loss = 0
            for batch in dataloader:
                s1, s2 = batch['sentence1'], batch['sentence2']
                target_sim = batch['score'].to(device, dtype=torch.float32)

                # 将文本送入模型所在的主设备 (例如 cuda:0)
                inputs1 = tokenizer(s1, padding=True, truncation=True, return_tensors="pt").to(llm_model.device)
                inputs2 = tokenizer(s2, padding=True, truncation=True, return_tensors="pt").to(llm_model.device)

                with torch.no_grad():
                    out_A = llm_model.model(**inputs1).last_hidden_state[:, -1, :].to(torch.float32).to(device)
                    out_B = llm_model.model(**inputs2).last_hidden_state[:, -1, :].to(torch.float32).to(device)

                sem_A = F.normalize(out_A, p=2, dim=-1)
                sem_B = F.normalize(out_B, p=2, dim=-1)

                msg = torch.sign(torch.randn(len(s1), msg_dim, 1)).to(device)
                msg[msg == 0] = 1.0

                optimizer.zero_grad()
                P_A = mlp_net(sem_A)
                P_B = mlp_net(sem_B)

                v_A = torch.bmm(P_A, msg).squeeze(-1) 
                v_B = torch.bmm(P_B, msg).squeeze(-1)

                # 确保 embeddings_weight 也在对应设备上
                emb_w = embeddings_weight.to(device)
                logits_A = torch.matmul(v_A, emb_w.T)
                logits_B = torch.matmul(v_B, emb_w.T)

                loss, _, _ = criterion(logits_A, logits_B, target_sim)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  -> Epoch {epoch+1:03d}/{EPOCHS} | Avg Loss: {total_loss/len(dataloader):.4f}")

        # 保存权重，命名格式与 run_experiments 严格对应
        save_path = f"core/context_aware_watermark_mlp_{msg_dim}b.pth"
        torch.save(mlp_net.state_dict(), save_path)
        print(f"[+] {msg_dim}-bit 模型训练完成并保存至: {save_path}")

if __name__ == "__main__":
    train()
