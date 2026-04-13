import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface_cache"

import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList
from datasets import load_dataset
import evaluate

from sentence_transformers import SentenceTransformer
from core.utils_crypto import RobustSemanticMLP
from core.hsp_processor import SemanticOrthogonalLogitsProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="HSP-Watermark Downstream Tasks Evaluation")
    # 标准任务专属模型
    parser.add_argument("--sum_model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--trans_model_name", type=str, default="facebook/mbart-large-50-many-to-many-mmt")

    parser.add_argument("--bit_dim", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)

    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=200)  # 摘要任务增加生成长度
    return parser.parse_args()


def load_metrics():
    print("正在加载评测指标 (ROUGE, BLEU, BERTScore)...")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    bertscore = evaluate.load("bertscore")
    return rouge, bleu, bertscore


def generate_text(model, tokenizer, prompt, max_new_tokens, processor=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    processors = LogitsProcessorList()
    if processor is not None:
        processors.append(processor)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=processors,
            do_sample=True,
            temperature=0.3,  # 任务微调模型降低随机性
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    # Seq2Seq模型直接取生成的token即可，无需减去prompt_len（因为decoder从0开始生成）
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def evaluate_summarization(model, tokenizer, wm_processor, metrics, num_samples, max_new_tokens):
    print("\n" + "="*50)
    print("文本摘要任务 | 模型: facebook/bart-large-cnn")
    print("="*50)
    rouge, _, bertscore = metrics

    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{num_samples}]")

    preds_clean = []
    preds_wm = []
    refs = []

    for item in tqdm(dataset):
        article = item["article"]
        # BART-Large-CNN是微调模型，直接输入文章即可，无需复杂提示词
        prompt = article[:1024]  # BART输入限制1024token
        reference = item["highlights"]

        gen_clean = generate_text(model, tokenizer, prompt, max_new_tokens, processor=None)
        gen_wm = generate_text(model, tokenizer, prompt, max_new_tokens, processor=wm_processor)

        preds_clean.append(gen_clean)
        preds_wm.append(gen_wm)
        refs.append(reference)

    rouge_clean = rouge.compute(predictions=preds_clean, references=refs)
    rouge_wm = rouge.compute(predictions=preds_wm, references=refs)

    bs_clean = bertscore.compute(predictions=preds_clean, references=refs, lang="en")
    bs_wm = bertscore.compute(predictions=preds_wm, references=refs, lang="en")

    clean_f1 = sum(bs_clean['f1']) / len(bs_clean['f1'])
    wm_f1 = sum(bs_wm['f1']) / len(bs_wm['f1'])

    print("\n[Summarization]")
    print(f"ROUGE-1      | 无水印: {rouge_clean['rouge1']:.4f} | 有水印: {rouge_wm['rouge1']:.4f}")
    print(f"ROUGE-2      | 无水印: {rouge_clean['rouge2']:.4f} | 有水印: {rouge_wm['rouge2']:.4f}")
    print(f"ROUGE-L      | 无水印: {rouge_clean['rougeL']:.4f} | 有水印: {rouge_wm['rougeL']:.4f}")
    print(f"BERTScore F1 | 无水印: {clean_f1:.4f} | 有水印: {wm_f1:.4f}")


def evaluate_translation(model, tokenizer, wm_processor, metrics, num_samples, max_new_tokens):
    print("\n" + "="*50)
    print("机器翻译任务 | 模型: facebook/mbart-large-50-many-to-many-mmt")
    print("="*50)
    _, bleu, bertscore = metrics

    dataset = load_dataset("wmt16", "ro-en", split=f"test[:{num_samples}]")

    preds_clean = []
    preds_wm = []
    refs = []

    # mBART-50需要指定语言代码
    tokenizer.src_lang = "ro_RO"
    tokenizer.tgt_lang = "en_XX"

    for item in tqdm(dataset):
        ro_text = item["translation"]["ro"]
        en_ref = item["translation"]["en"]
        
        # mBART微调模型直接输入源文本
        prompt = ro_text

        gen_clean = generate_text(model, tokenizer, prompt, max_new_tokens, processor=None)
        gen_wm = generate_text(model, tokenizer, prompt, max_new_tokens, processor=wm_processor)

        preds_clean.append(gen_clean)
        preds_wm.append(gen_wm)
        refs.append([en_ref])

    bleu_clean = bleu.compute(predictions=preds_clean, references=refs)
    bleu_wm = bleu.compute(predictions=preds_wm, references=refs)

    bs_clean = bertscore.compute(predictions=preds_clean, references=[r[0] for r in refs], lang="en")
    bs_wm = bertscore.compute(predictions=preds_wm, references=[r[0] for r in refs], lang="en")

    clean_f1 = sum(bs_clean['f1']) / len(bs_clean['f1'])
    wm_f1 = sum(bs_wm['f1']) / len(bs_wm['f1'])

    print("\n[Translation]")
    print(f"BLEU         | 无水印: {bleu_clean['score']:.4f} | 有水印: {bleu_wm['score']:.4f}")
    print(f"BERTScore F1 | 无水印: {clean_f1:.4f} | 有水印: {wm_f1:.4f}")


def create_watermark_processor(tokenizer, device, args):
    """为每个模型创建独立的水印处理器"""
    sent_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    mlp = RobustSemanticMLP(input_dim=384, hidden_dim=256, output_dim=128).to(device)

    weight_path = os.path.join("results", "robust_mlp.pth")
    if os.path.exists(weight_path):
        mlp.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

    secret_message = torch.sign(torch.randn(1, args.bit_dim)).to(device)
    secret_message[secret_message == 0] = 1.0

    return SemanticOrthogonalLogitsProcessor(
        sentence_model=sent_model,
        mlp_net=mlp,
        llm_tokenizer=tokenizer,
        message=secret_message,
        alpha=args.alpha,
        top_k=args.top_k
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = load_metrics()

    # ===================== 摘要模型 (BART-Large-CNN) =====================
    print("\n[1] 加载摘要模型")
    sum_tokenizer = AutoTokenizer.from_pretrained(args.sum_model_name)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(args.sum_model_name).to(device)
    sum_model.eval()

    sum_wm = create_watermark_processor(sum_tokenizer, device, args)
    evaluate_summarization(sum_model, sum_tokenizer, sum_wm, metrics, args.num_samples, args.max_new_tokens)

    # 释放显存
    del sum_model, sum_tokenizer, sum_wm
    torch.cuda.empty_cache()

    # ===================== 翻译模型 (mBART-Large-50) =====================
    print("\n[2] 加载翻译模型")
    trans_tokenizer = AutoTokenizer.from_pretrained(args.trans_model_name)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(args.trans_model_name).to(device)
    trans_model.eval()

    trans_wm = create_watermark_processor(trans_tokenizer, device, args)
    evaluate_translation(trans_model, trans_tokenizer, trans_wm, metrics, args.num_samples, args.max_new_tokens)


if __name__ == "__main__":
    main()
