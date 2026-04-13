import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 🔥 修复显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import math
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from sentence_transformers import SentenceTransformer
from core.utils_crypto import RobustSemanticMLP
from core.hsp_processor import SemanticOrthogonalLogitsProcessor
from core.hsp_detector import BlindSemanticDetector
from evaluation.attack_paraphrase import TextAttacker

def parse_args():
    parser = argparse.ArgumentParser(description="HSP-Watermark C4 Benchmark")
    parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/huggingface_models/Llama-2-7b-hf")
    parser.add_argument("--c4_path", type=str, default="./datasets/c4_subset_10000_15_20words_prompts.json")
    parser.add_argument("--output_log", type=str, default="results/experiment_log.jsonl")
    
    # --- 攻击配置 ---
    parser.add_argument("--attack_type", type=str, default="dipper", choices=["dipper", "pegasus", "none"], help="使用的改写模型类型")
    parser.add_argument("--paraphraser_path", type=str, default="/root/autodl-tmp/huggingface_models/kalpeshk2011-dipper-paraphraser-xxl", help="/root/autodl-tmp/huggingface_models/kalpeshk2011-dipper-paraphraser-xxl /root/models/pegasus-large")
    parser.add_argument("--drop_ratio", type=float, default=0.1, help="删词攻击的比例")
    
    # DIPPER 专用参数
    parser.add_argument("--lex_diversity", type=int, default=20, help="DIPPER: 词汇多样性 (0-100)")
    parser.add_argument("--order_diversity", type=int, default=0, help="DIPPER: 句法打乱程度 (0-100)")
    
    # Pegasus 专用参数
    parser.add_argument("--para_temperature", type=float, default=1.2, help="Pegasus: 改写采样温度")
    
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--bit_dim", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args()

def calculate_ppl(model, tokenizer, prompt_text, generated_text, device):
    encodings = tokenizer(generated_text, return_tensors="pt").to(device)
    prompt_encodings = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    prompt_len = prompt_encodings.input_ids.size(1)
    
    if input_ids.size(1) > prompt_len:
        target_ids[0, :prompt_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss
        return math.exp(loss.item())
    return float('nan')

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
            
    with open(args.c4_path, 'r', encoding='utf-8') as f:
        all_prompts = json.load(f)
    prompts = all_prompts[:args.num_samples]

    print("Loading Generative LLM...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # 🔥 修复3：Llama加载半精度，大幅节省显存
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    ).to(device)
    llm.eval()
    
    # 清理显存缓存
    torch.cuda.empty_cache()
    
    # 初始化独立的攻击模块
    attacker = TextAttacker(
        model_name=args.paraphraser_path if args.attack_type != "none" else None, 
        attack_type=args.attack_type, 
        device=device
    )

    print("Loading Feature Extractors...")
    sent_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    mlp = RobustSemanticMLP(input_dim=384, hidden_dim=256, output_dim=128).to(device)
    
    weight_path = os.path.join("results", "robust_mlp.pth")
    if os.path.exists(weight_path):
        mlp.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

    secret_message = torch.sign(torch.randn(1, args.bit_dim)).to(device)
    secret_message[secret_message == 0] = 1.0
    bits_for_log = [1 if x > 0 else 0 for x in secret_message.tolist()[0]]

    detector = BlindSemanticDetector(
        sentence_model=sent_model,
        mlp_net=mlp,
        llm_tokenizer=tokenizer,
        message_dim=args.bit_dim
    )

    total_acc_clean = 0.0
    total_acc_drop = 0.0
    total_acc_para = 0.0
    total_ppl_wm, total_ppl_base, valid_ppl_count = 0.0, 0.0, 0

    # ===================== 新增：有效样本计数 =====================
    valid_sample_count = 0

    for i, prompt in enumerate(tqdm(prompts, desc=f"Evaluating {args.bit_dim}-bits")):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_token_len = inputs.input_ids.size(1)  # ✅ prompt 的 token 数
        
        # [A] 生成 Baseline
        with torch.no_grad():
            outputs_base = llm.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, top_p=args.top_p, temperature=args.temperature, pad_token_id=tokenizer.eos_token_id)
            base_full = tokenizer.decode(outputs_base[0], skip_special_tokens=True)
            
        # [B] 生成 Watermarked
        watermark_processor = SemanticOrthogonalLogitsProcessor(sent_model, mlp, tokenizer, secret_message, args.alpha, args.top_k)
        with torch.no_grad():
            outputs_wm = llm.generate(**inputs, max_new_tokens=args.max_new_tokens, logits_processor=LogitsProcessorList([watermark_processor]), do_sample=True, top_p=args.top_p, temperature=args.temperature, pad_token_id=tokenizer.eos_token_id)
            wm_full = tokenizer.decode(outputs_wm[0], skip_special_tokens=True)

        # ===================== 核心修改 =====================
        # ✅ 计算实际生成的 token 数量
        actual_generated_tokens = outputs_wm.size(1) - prompt_token_len

        # ✅ 判断是否满足最小长度要求
        min_required_tokens = int(args.max_new_tokens * 0.75)
        is_valid_sample = actual_generated_tokens >= min_required_tokens

        # ===================== 提取文本 =====================
        clean_text = tokenizer.decode(outputs_base[0][prompt_token_len:], skip_special_tokens=True)
        wm_text = tokenizer.decode(outputs_wm[0][prompt_token_len:], skip_special_tokens=True)

        # ===================== 初始化指标 =====================
        acc_clean = 0.0
        acc_drop = 0.0
        acc_para = 0.0
        ppl_base = float('nan')
        ppl_wm = float('nan')
        dropped_text = ""
        paraphrased_text = ""

        # ===================== 只有有效样本才计算指标 =====================
        if is_valid_sample:
            # [C] 提取 Clean 准确率与计算 PPL
            acc_clean = (detector.extract_message(wm_full) == secret_message).sum().item() / args.bit_dim
            total_acc_clean += acc_clean
            
            ppl_base = calculate_ppl(llm, tokenizer, prompt, base_full, device)
            ppl_wm = calculate_ppl(llm, tokenizer, prompt, wm_full, device)
            if not math.isnan(ppl_base) and not math.isnan(ppl_wm):
                total_ppl_base += ppl_base
                total_ppl_wm += ppl_wm
                valid_ppl_count += 1

            # [D] Drop 攻击
            dropped_text = attacker.simulate_drop_attack(wm_text, drop_ratio=args.drop_ratio)
            acc_drop = (detector.extract_message(prompt + " " + dropped_text) == secret_message).sum().item() / args.bit_dim
            total_acc_drop += acc_drop

            # [E] Paraphrase 改写攻击
            paraphrased_text = attacker.paraphrase(
                wm_text, 
                lex_diversity=args.lex_diversity, 
                order_diversity=args.order_diversity,
                temperature=args.para_temperature
            )
            
            if paraphrased_text:
                acc_para = (detector.extract_message(prompt + " " + paraphrased_text) == secret_message).sum().item() / args.bit_dim
            total_acc_para += acc_para

            # 累计有效样本数
            valid_sample_count += 1

        # ===================== 保存日志（永远记录，包含无效样本） =====================
        log_record = {
            "sample_id": i + 1,
            "prompt": prompt,
            "original_bits": bits_for_log,
            "clean_text": clean_text,
            "watermarked_text": wm_text,
            "paraphrased_text": paraphrased_text,
            
            # ✅ 新增：实际生成的 token 数
            "actual_generated_tokens": actual_generated_tokens,
            "max_new_tokens": args.max_new_tokens,
            "min_required_tokens": min_required_tokens,
            "is_valid_sample": is_valid_sample,  # ✅ 是否有效

            "clean_ppl": round(ppl_base, 4) if not math.isnan(ppl_base) else 0.0,
            "wm_ppl": round(ppl_wm, 4) if not math.isnan(ppl_wm) else 0.0,
            "bit_acc_clean": round(acc_clean, 4),
            "bit_acc_drop30": round(acc_drop, 4),
            "bit_acc_paraphrase": round(acc_para, 4)
        }
        with open(args.output_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + '\n')

    # ===================== 最终输出：只统计有效样本 =====================
    v = valid_sample_count if valid_sample_count > 0 else 1
    print("\n" + "="*80)
    print(f"🌟 RESULTS (Attack: {args.attack_type.upper()}) 🌟")
    print(f"有效样本数 / 总样本数: {valid_sample_count} / {len(prompts)}")
    print(f"Clean Bit Acc: {total_acc_clean/v*100:.2f}% | Drop Acc: {total_acc_drop/v*100:.2f}% | Para Acc: {total_acc_para/v*100:.2f}%")
    print(f"Base PPL: {total_ppl_base/v:.2f} | WM PPL: {total_ppl_wm/v:.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
