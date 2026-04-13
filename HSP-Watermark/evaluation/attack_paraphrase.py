import torch
import random
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

class TextAttacker:
    def __init__(self, model_name=None, attack_type="dipper", device="cuda"):
        self.device = device
        self.attack_type = attack_type.lower()
        self.model = None
        self.tokenizer = None

        if self.attack_type in ["dipper", "pegasus"] and model_name:
            print(f"Loading {self.attack_type.capitalize()} paraphraser from {model_name}...")
            
            # T5 专用分词器
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                # device_map="auto",  # 自动平衡，不爆显存
                tie_word_embeddings=False,
                local_files_only=True # 禁止联网，加速加载
            )
            self.model.eval()

    def simulate_drop_attack(self, text: str, drop_ratio: float = 0.3) -> str:
        words = text.split()
        if not words: return text
        keep_count = int(len(words) * (1 - drop_ratio))
        if keep_count == 0: return text
        keep_indices = sorted(random.sample(range(len(words)), keep_count))
        return " ".join([words[i] for i in keep_indices])

    def paraphrase(self, text: str, **kwargs) -> str:
        if not text.strip() or self.model is None:
            return text

        if self.attack_type == "dipper":
            return self._dipper_paraphrase(text, **kwargs)
        elif self.attack_type == "pegasus":
            return self._pegasus_paraphrase(text, **kwargs)
        return text

    def _dipper_paraphrase(self, text: str, lex_diversity=60, order_diversity=0, **kwargs) -> str:
        prompt = f"lexical = {lex_diversity}, order = {order_diversity} <sent> {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        # 自动匹配设备
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                do_sample=True,
                top_p=0.75,
                top_k=0
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _pegasus_paraphrase(self, text: str, temperature=1.2, num_beams=5, **kwargs) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=num_beams,
                temperature=temperature
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
