import time
import torch
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize

class TextAttacker:
    def __init__(self, model_name=None, attack_type="dipper", device="cuda"):
        self.device = device
        self.attack_type = attack_type.lower()
        self.model = None
        self.tokenizer = None

        if self.attack_type == "dipper" and model_name:
            print(f"Loading Dipper model from {model_name}...")
            import time
            time1 = time.time()
            self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, 
                device_map='auto',
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
            print(f"Dipper model loaded in {time.time() - time1:.2f}s")
            self.model.eval()
            
        elif self.attack_type == "pegasus" and model_name:
            print(f"Loading Pegasus model from {model_name}...")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                tie_word_embeddings=False,
                local_files_only=True
            )
            self.model.eval()
    def simulate_drop_attack(self, text: str, drop_ratio: float = 0.3) -> str:
        """
        随机丢弃攻击（Deletion Attack）
        """
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

    def _dipper_paraphrase(self, text: str, lex_diversity=20, order_diversity=0, sent_interval=1, prefix="", **kwargs) -> str:
        """
        完全对齐基准论文的 Dipper 改写代码
        加入超长前缀截断以防止 >512 Token 的报错
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)
        print(f"lex_code: {lex_code} | order_code: {order_code}")
        # 对齐基准：标准清洗和 NLTK 切分
        text = " ".join(text.split())
        sentences = sent_tokenize(text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""
        
        # 默认参数对齐基准要求
        if 'do_sample' not in kwargs:
            kwargs['do_sample'] = True
        if 'top_p' not in kwargs:
            kwargs['top_p'] = 0.75
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            
            # 🚀 保护机制：限制 prefix 长度，防止长文本累加爆掉 T5 的 512 上限
            prefix_words = prefix.split()
            if len(prefix_words) > 100:
                prefix = " ".join(prefix_words[-100:])
            
            # 严格对齐基准：拼接格式
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            # 加上 truncation=True 作为最后一道保险
            final_input = self.tokenizer(
                [final_input_text], 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            final_input = {k: v.to(self.model.device) for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
                
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 严格对齐基准：直接累加输出
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text.strip()

    def _pegasus_paraphrase(self, text: str, temperature=1.2, num_beams=5, **kwargs) -> str:
        """
        Pegasus 模型改写攻击
        """
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
