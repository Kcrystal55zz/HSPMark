import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextAttacker:
    def __init__(self, model_name=None, attack_type="dipper", device="cuda"):
        """
        初始化攻击器
        :param model_name: 攻击模型的本地路径或HF仓库名
        :param attack_type: "dipper" 或 "pegasus" 或 "none"
        """
        self.device = device
        self.attack_type = attack_type.lower()
        self.model = None
        self.tokenizer = None

        if self.attack_type in ["dipper", "pegasus"] and model_name:
            print(f"Loading {self.attack_type.capitalize()} paraphraser from {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()

    def simulate_drop_attack(self, text: str, drop_ratio: float = 0.3) -> str:
        """随机删除指定比例的单词（词级别 Drop）"""
        words = text.split()
        if not words: return text
        keep_count = int(len(words) * (1 - drop_ratio))
        if keep_count == 0: return text
        
        keep_indices = sorted(random.sample(range(len(words)), keep_count))
        return " ".join([words[i] for i in keep_indices])

    def paraphrase(self, text: str, **kwargs) -> str:
        """统一的改写接口"""
        if not text.strip() or self.model is None:
            return text

        if self.attack_type == "dipper":
            return self._dipper_paraphrase(text, **kwargs)
        elif self.attack_type == "pegasus":
            return self._pegasus_paraphrase(text, **kwargs)
        else:
            return text

    def _dipper_paraphrase(self, text: str, lex_diversity=60, order_diversity=0, **kwargs) -> str:
        """
        DIPPER 改写逻辑
        DIPPER 使用特殊的 prompt 格式来控制改写强度
        例如: "lexical = 60, order = 0 <sent> 原始文本"
        """
        prompt = f"lexical = {lex_diversity}, order = {order_diversity} <sent> {text}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                do_sample=True, # DIPPER 推荐使用采样
                top_p=0.75,
                top_k=0
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _pegasus_paraphrase(self, text: str, temperature=1.2, num_beams=5, **kwargs) -> str:
        """
        Pegasus 常规改写逻辑
        依赖 beam search 和 temperature 来增加多样性
        """
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                num_beams=num_beams,
                temperature=temperature
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
