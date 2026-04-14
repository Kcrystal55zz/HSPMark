import torch
import random
import re
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
                device_map="auto",  # 自动平衡，不爆显存
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
    def _dipper_paraphrase(self, text: str, lex_diversity=20, order_diversity=0, sent_interval=1, **kwargs) -> str:
        """
        修正版 DIPPER 改写，吸收了官方的 Prompt 格式和 100-diversity 逻辑。
        sent_interval: 控制每次改写的句子数量（默认为 1，即严格单句改写）
        """
        # 1. 强制参数对齐原论文 (必须是 0, 20, 40, 60, 80, 100)
        # DIPPER 模型内部只认识这些离散的值
        valid_diversities = [0, 20, 40, 60, 80, 100]
        lex_diversity = min(valid_diversities, key=lambda x: abs(x - lex_diversity))
        order_diversity = min(valid_diversities, key=lambda x: abs(x - order_diversity))

        # 2. 最关键的 Bug 修复：参数翻转
        lex_code = 100 - lex_diversity
        order_code = 100 - order_diversity
        print(f"lex_code / order_code: {lex_code} / {order_code}")
        if sent_interval <= 0:
            chunks = [text]
        else:
            try:
                import nltk
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(text)
            except Exception:
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            for i in range(0, len(sentences), sent_interval):
                chunk = " ".join(sentences[i:i + sent_interval])
                if chunk.strip():
                    chunks.append(chunk)

        paraphrased_chunks = []
        
        # 3. 引入 Context Prefix 机制 (让上一句指导下一句的改写，保留上下文连贯性)
        prefix = "" 

        for chunk in chunks:
            # 拼接正确的 Prompt，加上 </sent> 闭合标签
            prompt = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                prompt += f" {prefix}"
            prompt += f" <sent> {chunk} </sent>"

            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    do_sample=True,
                    top_p=0.75,
                    top_k=0
                )
            decoded_chunk = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # 后处理：有时 DIPPER 会把输入的前缀也重复吐出来，我们需要确保只保留真正改写的部分
            # (实践中 DIPPER 倾向于输出干净的改写句，但做个防卫)
            if prefix and decoded_chunk.startswith(prefix):
                decoded_chunk = decoded_chunk[len(prefix):].strip()

            paraphrased_chunks.append(decoded_chunk)
            
            # 将当前未改写的原文（或改写后的句子）作为下一个块的 Prefix
            # 这里我们用原文作为 prefix 以防止错误累加
            prefix = chunk 

        return " ".join(paraphrased_chunks)

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
