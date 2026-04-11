
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ParaphraseAttacker:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        基于本地 Seq2Seq 模型（如 DIPPER, PEGASUS）的重写攻击器。
        
        :param model_path: 本地模型权重路径或 HuggingFace 仓名
        :param device: 运行设备
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 🚨 关键修改：强行使用 fp16 半精度加载，并将模型通过 device_map 切分到各个显卡上。
        # 否则加载 DIPPER XXL (11B) 会直接撑爆绝大多数单卡显存。
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def attack(self, text: str, max_length: int = 512, **generation_kwargs) -> str:
        """
        执行语义重写攻击。
        
        :param text: 待重写的原始生成文本
        :param max_length: 截断长度
        :return: 重写后的文本
        """
        # Dipper 和 PEGASUS 往往只需要直接输入英文文本，不需要前缀指令（如 "paraphrase: "）
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        ).to(self.device)

        gen_params = {
            "max_length": max_length,
            "num_beams": 4,          # 使用集束搜索，让重写质量更高（攻击力更强）
            "early_stopping": True
        }
        gen_params.update(generation_kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)
            
        paraphrased_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return paraphrased_text
