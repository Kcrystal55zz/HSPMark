import torch
import re
from .utils_crypto import get_seed_from_tensor, generate_orthogonal_matrix

class BlindSemanticDetector:
    def __init__(self, sentence_model, mlp_net, llm_tokenizer, message_dim: int):
        self.sentence_model = sentence_model
        self.mlp_net = mlp_net.eval()
        self.tokenizer = llm_tokenizer
        self.message_dim = message_dim
        self.vocab_size = len(llm_tokenizer)

    def extract_message(self, text: str) -> torch.Tensor:
        device = next(self.mlp_net.parameters()).device
        
        sentences = re.split(r'([.!?\n]+)', text)
        combined_sentences = []
        for i in range(0, len(sentences)-1, 2):
            s = sentences[i] + sentences[i+1]
            if len(s.strip()) > 5:
                combined_sentences.append(s.strip())
        if len(sentences) % 2 != 0 and len(sentences[-1].strip()) > 5:
            combined_sentences.append(sentences[-1].strip())

        if len(combined_sentences) < 2:
            return torch.ones(1, self.message_dim).to(device)

        score_accumulator = torch.zeros(1, self.message_dim).to(device)
        valid_tokens_count = 0

        with torch.no_grad():
            for i in range(1, len(combined_sentences)):
                anchor_sentence = combined_sentences[i-1]
                target_sentence = combined_sentences[i]

                # 确保提取种子的锚点和生成时一模一样
                sent_embed = self.sentence_model.encode(anchor_sentence, convert_to_tensor=True)
                sent_embed = sent_embed.to(device).unsqueeze(0)
                robust_seed_tensor = self.mlp_net(sent_embed)
                seed = get_seed_from_tensor(robust_seed_tensor)

                U_i = generate_orthogonal_matrix(seed, self.message_dim, self.vocab_size, device)

                # 修复: 补回前导空格，模拟流式生成时的 BPE 分词行为
                prefix_target = " " + target_sentence if not target_sentence.startswith(" ") else target_sentence
                target_ids = self.tokenizer.encode(prefix_target, add_special_tokens=False)

                for token_id in target_ids:
                    token_vector = U_i[:, token_id].unsqueeze(0)
                    score_accumulator += token_vector
                    valid_tokens_count += 1

        if valid_tokens_count == 0:
            return torch.ones(1, self.message_dim).to(device)

        extracted_message = torch.sign(score_accumulator)
        extracted_message[extracted_message == 0] = 1.0 
        return extracted_message
