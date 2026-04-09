import os
# ==============================================================================
# 🚨 终极环境修复
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time
import numpy as np
import argparse
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LogitsProcessorList, MBartForConditionalGeneration
from sklearn.metrics import roc_curve, auc
import evaluate

transformers.logging.set_verbosity_error()

from core.utils_crypto import generate_private_key, encode_message, decode_message
from core.hsp_processor import HSPWatermarkLogitsProcessor
from core.hsp_detector import HSPWatermarkDetector
from evaluation.attack_perturb import PerturbationAttacker
from evaluation.attack_paraphrase import ParaphraseAttacker

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "models": {
        "generation": "/root/autodl-tmp/huggingface_models/Llama-2-7b-hf",                 
        "summarization": "/root/models/bart-large",               
        "translation": "/root/models/mbart-large-50",
        "paraphrase_dipper": "kalpeshk2011/dipper-paraphraser-xxl", 
        "paraphrase_pegasus": "/root/models/pegasus-large"         
    },
    "datasets": {
        "generation": "./datasets/c4_subset_10000_15_20words_prompts.json",     
        "summarization": "./datasets/cnn_articles.json",
        "translation": "./datasets/wmt_en.json"         
    },
    "capacities": [24, 36], 
    "alpha": 2.0
}

class MetricsEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        print("[*] 正在加载评测指标 (Rouge, BERTScore, BLEU)...")
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')
        self.bleu = evaluate.load('sacrebleu')

    def calc_ppl(self, model, input_ids):
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()

    def calc_text_quality(self, preds: list, refs: list, task: str):
        results = {}
        if task == "summarization":
            rouge_out = self.rouge.compute(predictions=preds, references=refs)
            results['ROUGE-1'] = rouge_out['rouge1']
        elif task == "translation":
            bleu_out = self.bleu.compute(predictions=preds, references=refs)
            results['BLEU'] = bleu_out['score']
        bs_out = self.bertscore.compute(predictions=preds, references=refs, lang="en")
        results['BERTScore'] = np.mean(bs_out['f1'])
        return results

def compute_detection_metrics(watermarked_scores: list, clean_scores: list):
    y_true = [1] * len(watermarked_scores) + [0] * len(clean_scores)
    y_scores = watermarked_scores + clean_scores
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_1_fpr = tpr[np.where(fpr <= 0.01)[0][-1]] if len(np.where(fpr <= 0.01)[0]) > 0 else 0.0
    return {"AUC": roc_auc, "TPR@1%FPR": tpr_at_1_fpr}

def prepare_local_prompts(task: str, num_samples: int, tokenizer, max_prompt_length: int = 50) -> list:
    prompts = []
    file_path = CONFIG["datasets"].get(task, "")
    print(f"[*] 正在尝试读取本地数据集文件: {file_path} ...")
    
    if not os.path.exists(file_path):
        print(f"[!] 警告: 找不到本地文件 '{file_path}'！将使用内置备用提示词...")
        fallback_prompts = [
            "After the martyrdom of St. Boniface, Vergilius was made Bishop of Salzburg",
            "\"Whoever gets him, they'll be getting a good one,\" David Montgomery said."
        ]
        return (fallback_prompts * (num_samples // len(fallback_prompts) + 1))[:num_samples]

    try:
        if file_path.endswith('.json'):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                    prompts = data[:num_samples]
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    for item in data:
                        text = item.get("text", item.get("article", item.get("en", item.get("prompt", ""))))
                        if text: prompts.append(text)
                        if len(prompts) >= num_samples: break
                elif isinstance(data, dict):
                    for val in data.values():
                        if isinstance(val, list):
                            for item in val:
                                text = item if isinstance(item, str) else item.get("text", "")
                                if text: prompts.append(text)
                                if len(prompts) >= num_samples: break
                            break
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line: prompts.append(line)
                    if len(prompts) >= num_samples: break

        processed_prompts = []
        for text in prompts:
            if not text: continue
            if task == "generation":
                tokens = tokenizer.encode(text)[:max_prompt_length]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            processed_prompts.append(text)
            if len(processed_prompts) >= num_samples: break
        prompts = processed_prompts

    except Exception as e:
        print(f"[!] 读取本地 JSON 文件时发生错误: {e}")
        prompts = []

    if len(prompts) == 0:
        print("[!] 文件为空或解析失败！使用内置备用提示词。")
        prompts = ["Test prompt 1", "Test prompt 2"]
        
    print(f"[+] 数据集准备完毕！共获取 {len(prompts)} 条有效提示词。")
    if len(prompts) < num_samples:
        print(f"[*] 提示：本地数据不足 {num_samples} 条，将循环使用以满足测试数量。")
        prompts = (prompts * (num_samples // len(prompts) + 1))[:num_samples]
        
    return prompts

def run_evaluation_pipeline(args):
    os.makedirs("results", exist_ok=True)
    
    evaluator = MetricsEvaluator(device=CONFIG["device"])
    perturb_attacker = PerturbationAttacker(seed=42)
    
    para_attacker = None
    if args.paraphrase_model != "none":
        model_key = f"paraphrase_{args.paraphrase_model}"
        print(f"[*] 正在加载 {args.paraphrase_model.upper()} 重写模型...")
        para_attacker = ParaphraseAttacker(model_path=CONFIG["models"][model_key], device=CONFIG["device"])
    
    tasks_to_run = ["generation", "summarization", "translation"] if args.task == "all" else [args.task]

    for task in tasks_to_run:
        print(f"\n========== 初始化任务: {task.upper()} ==========")
        model_path = CONFIG["models"][task]
        
        print(f"[*] 正在加载主模型: {model_path} ...")
        if task == "generation":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
            embeddings_weight = model.get_output_embeddings().weight.detach()
        elif task == "translation":
            tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="en_XX")
            model = MBartForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
            embeddings_weight = model.get_output_embeddings().weight.detach()
        else: # summarization
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
            embeddings_weight = model.get_output_embeddings().weight.detach()
            
        model.eval()
        
        dataset_prompts = prepare_local_prompts(task, args.num_samples, tokenizer)
        
        # 设定有效样本过滤阈值（例如要求生成长度大于等于目标长度的 75%）
        filter_threshold = int(args.token_length * 0.75) 

        for capacity in CONFIG["capacities"]:
            p_matrix = generate_private_key(embeddings_weight.shape[1], capacity, device=CONFIG["device"])
            p_matrix = p_matrix.to(embeddings_weight.dtype)
            detector = HSPWatermarkDetector(embeddings_weight, p_matrix)
            
            print(f"\n--- [Task: {task} | Cap: {capacity}bits | Target Length: {args.token_length}tokens] ---")
            result_file = f"results/exp_{task}_{capacity}b_{args.token_length}t.jsonl"
            print(f"[*] 详细数据将实时保存至: {result_file}")
            print(f"[*] 注意: 最终计算平均指标时，将只统计生成真实 Token 数量 >= {filter_threshold} 的有效样本。")
            
            # 用于记录所有明细的数组
            all_results = []
            
            with open(result_file, "w", encoding="utf-8") as f_out:
                for idx, prompt in enumerate(dataset_prompts):
                    print(f"  -> [Sample {idx+1}/{len(dataset_prompts)}] Generating & Attacking...")
                    
                    original_bits = torch.randint(0, 2, (capacity,)).tolist()
                    msg_tensor = encode_message(original_bits, device=CONFIG["device"])
                    msg_tensor = msg_tensor.to(embeddings_weight.dtype)
                    
                    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(CONFIG["device"])
                    
                    # 🌟 修复: 移除强迫模型生成废话的 min_new_tokens，允许模型自然结束
                    gen_kwargs = {
                        "max_new_tokens": args.token_length, 
                        "do_sample": True, 
                        "top_p": 0.9,
                        "temperature": 0.8,
                        "repetition_penalty": 1.2, # 保持重复惩罚
                        "pad_token_id": tokenizer.eos_token_id
                    }
                    if task == "translation":
                        gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["ro_RO"]
                    
                    with torch.no_grad():
                        clean_out = model.generate(input_ids, **gen_kwargs)
                        clean_ids = clean_out[0, input_ids.shape[1]:] if task == "generation" else clean_out[0, 1:]
                        clean_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
                        _, clean_score = detector.detect(clean_ids)
                        clean_detect_score_val = torch.norm(clean_score).item()
                    
                    hsp_processor = HSPWatermarkLogitsProcessor(embeddings_weight, msg_tensor, p_matrix, alpha=CONFIG["alpha"])
                    gen_kwargs["logits_processor"] = LogitsProcessorList([hsp_processor])
                    with torch.no_grad():
                        wm_out = model.generate(input_ids, **gen_kwargs)
                        wm_ids = wm_out[0, input_ids.shape[1]:] if task == "generation" else wm_out[0, 1:]
                        wm_text = tokenizer.decode(wm_ids, skip_special_tokens=True)
                        
                    clean_ppl_val, wm_ppl_val = 0.0, 0.0
                    if task == "generation":
                        clean_ppl_val = evaluator.calc_ppl(model, clean_out)
                        wm_ppl_val = evaluator.calc_ppl(model, wm_out)
                        
                    extracted_msg, wm_score = detector.detect(wm_ids)
                    wm_detect_score_val = torch.norm(wm_score).item()
                    
                    extracted_bits = decode_message(extracted_msg)
                    bit_acc_clean = 1.0 - (sum([1 for e, o in zip(extracted_bits, original_bits) if e != o]) / capacity)
                    
                    crop_ids = perturb_attacker.attack_drop(wm_ids.tolist(), drop_ratio=0.3)
                    crop_extracted, _ = detector.detect(torch.tensor(crop_ids, device=CONFIG["device"]))
                    bit_acc_crop = 1.0 - (sum([1 for e, o in zip(decode_message(crop_extracted), original_bits) if e != o]) / capacity)

                    bit_acc_para = 0.0
                    para_text = ""
                    if para_attacker is not None:
                        para_text = para_attacker.attack(wm_text)
                        para_ids = tokenizer(para_text, return_tensors="pt").input_ids[0].to(CONFIG["device"])
                        para_extracted, _ = detector.detect(para_ids)
                        bit_acc_para = 1.0 - (sum([1 for e, o in zip(decode_message(para_extracted), original_bits) if e != o]) / capacity)

                    actual_generated_tokens = len(wm_ids)
                    
                    record = {
                        "sample_id": idx + 1,
                        "prompt": prompt,
                        "original_bits": original_bits,
                        "clean_text": clean_text,
                        "watermarked_text": wm_text,
                        "paraphrased_text": para_text,
                        "generated_tokens": actual_generated_tokens, # 这是真实的生成长度
                        "clean_ppl": round(clean_ppl_val, 4),
                        "wm_ppl": round(wm_ppl_val, 4),
                        "clean_detect_score": round(clean_detect_score_val, 4),
                        "wm_detect_score": round(wm_detect_score_val, 4),
                        "bit_acc_clean": bit_acc_clean,
                        "bit_acc_drop30": bit_acc_crop,
                        "bit_acc_paraphrase": bit_acc_para
                    }
                    all_results.append(record)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()

            # 🌟 修复: 筛选有效样本计算平均值
            valid_results = [r for r in all_results if r["generated_tokens"] >= filter_threshold]
            num_valid = len(valid_results)
            
            print(f"\n=== [全局指标汇总 Task: {task} | 有效样本: {num_valid}/{len(all_results)}] ===")
            
            if num_valid == 0:
                print("    [!] 警告：没有达到长度阈值的样本，无法计算有效平均值。")
                continue
                
            wm_detect_scores = [r["wm_detect_score"] for r in valid_results]
            clean_detect_scores = [r["clean_detect_score"] for r in valid_results]
            bit_acc_clean_list = [r["bit_acc_clean"] for r in valid_results]
            bit_acc_copy_list = [r["bit_acc_drop30"] for r in valid_results]
            bit_acc_para_list = [r["bit_acc_paraphrase"] for r in valid_results]
            
            if task == "generation":
                clean_ppl_list = [r["clean_ppl"] for r in valid_results]
                wm_ppl_list = [r["wm_ppl"] for r in valid_results]
                print(f"    [文本质量] Clean PPL: {np.mean(clean_ppl_list):.4f} | Watermarked PPL: {np.mean(wm_ppl_list):.4f}")
            else:
                wm_texts = [r["watermarked_text"] for r in valid_results]
                clean_texts = [r["clean_text"] for r in valid_results]
                quality = evaluator.calc_text_quality(wm_texts, clean_texts, task)
                print(f"    [文本质量] " + ", ".join([f"{k}: {v:.4f}" for k, v in quality.items()]))
            
            det_metrics = compute_detection_metrics(wm_detect_scores, clean_detect_scores)
            print(f"    [检测能力] AUC: {det_metrics['AUC']: .4f} | TPR@1%FPR: {det_metrics['TPR@1%FPR']: .4f}")
            
            res_str = f"    [平均鲁棒性] Clean Acc: {np.mean(bit_acc_clean_list)*100:.2f}% | Drop Acc: {np.mean(bit_acc_copy_list)*100:.2f}%"
            if para_attacker is not None:
                res_str += f" | Paraphrase Acc: {np.mean(bit_acc_para_list)*100:.2f}%"
            print(res_str)
                
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSP-Watermark Top-Tier Evaluation Engine")
    parser.add_argument("--task", type=str, default="generation", choices=["generation", "summarization", "translation", "all"])
    parser.add_argument("--paraphrase_model", type=str, default="none", choices=["none", "dipper", "pegasus"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--token_length", type=int, default=200)
    args = parser.parse_args()
    run_evaluation_pipeline(args)