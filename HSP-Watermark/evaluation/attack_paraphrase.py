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
        valid_diversities = [0, 20, 40, 60, 80, 100]
        lex_diversity = min(valid_diversities, key=lambda x: abs(x - lex_diversity))
        order_diversity = min(valid_diversities, key=lambda x: abs(x - order_diversity))

        lex_code = 100 - lex_diversity
        order_code = 100 - order_diversity
        
        if sent_interval <= 0:
            chunks = [text]
        else:
            # 🚀 修复1：使用和 hsp_processor.py / hsp_detector.py 完全一模一样的正则切分！
            # 绝对不要用 NLTK，保证生成、攻击、检测三端切分逻辑 100% 咬合
            import re
            raw_sentences = re.split(r'([.!?\n]+)', text)
            sentences = []
            for i in range(0, len(raw_sentences)-1, 2):
                s = raw_sentences[i] + raw_sentences[i+1]
                if len(s.strip()) > 0:
                    sentences.append(s.strip())
            if len(raw_sentences) % 2 != 0 and len(raw_sentences[-1].strip()) > 0:
                sentences.append(raw_sentences[-1].strip())
            
            chunks = []
            for i in range(0, len(sentences), sent_interval):
                chunk = " ".join(sentences[i:i + sent_interval])
                if chunk.strip():
                    chunks.append(chunk)

        paraphrased_chunks = []
        prefix = "" 

        for chunk in chunks:
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
            
            # 🚀 修复3：更鲁棒的前缀剥离（不管大小写和多余空格，只要前缀长度匹配就切除）
            if prefix:
                # 因为大模型可能改变标点或大小写，使用长度估算来剥离幻觉复述的前缀
                if len(decoded_chunk) > len(prefix) and decoded_chunk[:len(prefix)//2].lower() == prefix[:len(prefix)//2].lower():
                    decoded_chunk = decoded_chunk[len(prefix):].strip()

            # 🚀 修复2：防止 DIPPER 吞标点导致检测端句子合并
            if decoded_chunk and decoded_chunk[-1] not in ['.', '!', '?']:
                # 寻找原文的标点，如果原文有标点，给它补上
                last_char = chunk.strip()[-1] if chunk.strip() else '.'
                if last_char in ['.', '!', '?']:
                    decoded_chunk += last_char
                else:
                    decoded_chunk += '.'

            paraphrased_chunks.append(decoded_chunk)
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
