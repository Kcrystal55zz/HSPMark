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
import torch.nn.functional as F

transformers.logging.set_verbosity_error()

# 引入全新的上下文感知动态水印模块
from core.hsp_processor import ContextAwareHSPLogitsProcessor
from core.hsp_detector import ContextAwareHSPDetector
from core.train_mlp_with_llm import ContextAwareWatermarkNet

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
    "capacities": [16, 24], # 测试的比特容量，需对应训练好不同的 MLP 权重
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
        
        # 确定底层 Backbone (用于提取 Hidden States)
        llm_backbone = getattr(model, "model", getattr(model, "transformer", getattr(model, "get_decoder", lambda: model)()))
        
        # 统一把词表转为 float32
        embeddings_weight = F.normalize(embeddings_weight, p=2, dim=-1).to(torch.float32)
        HIDDEN_DIM = embeddings_weight.shape[1]
        SEMANTIC_DIM = HIDDEN_DIM
        
        dataset_prompts = prepare_local_prompts(task, args.num_samples, tokenizer)
        filter_threshold = int(args.token_length * 0.75) 

        for capacity in CONFIG["capacities"]:
            # 【重要修改】：不同 capacity 需要加载其对应的 MLP 权重
            mlp_net = ContextAwareWatermarkNet(SEMANTIC_DIM, capacity, HIDDEN_DIM).to(CONFIG["device"])
            weight_path = f"core/context_aware_watermark_mlp_{capacity}b.pth"
            if os.path.exists(weight_path):
                mlp_net.load_state_dict(torch.load(weight_path, map_location=CONFIG["device"]))
                print(f"[+] 成功加载水印网络权重: {weight_path}")
            else:
                print(f"[!] 警告: 未找到 {weight_path}，使用随机初始化权重（仅作代码演示）")
                
            detector = ContextAwareHSPDetector(mlp_net, llm_backbone, tokenizer, embeddings_weight)
            
            print(f"\n--- [Task: {task} | Cap: {capacity}bits | Target Length: {args.token_length}tokens] ---")
            result_file = f"results/exp_{task}_{capacity}b_{args.token_length}t.jsonl"
            print(f"[*] 详细数据将实时保存至: {result_file}")
            
            all_results = []
            
            with open(result_file, "w", encoding="utf-8") as f_out:
                for idx, prompt in enumerate(dataset_prompts):
                    print(f"  -> [Sample {idx+1}/{len(dataset_prompts)}] Generating & Attacking...")
                    
                    # 1. 随机生成 0/1 数组，并转化为 MLP 需要的 [-1, 1] 浮点张量
                    original_bits = torch.randint(0, 2, (capacity,)).tolist()
                    msg_tensor = torch.tensor([1.0 if b == 1 else -1.0 for b in original_bits], device=CONFIG["device"], dtype=torch.float32)
                    
                    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(CONFIG["device"])
                    
                    gen_kwargs = {
                        "max_new_tokens": args.token_length, 
                        "do_sample": True, 
                        "top_p": 0.9,
                        "temperature": 0.8,
                        "repetition_penalty": 1.2, 
                        "pad_token_id": tokenizer.eos_token_id
                    }
                    if task == "translation":
                        gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["ro_RO"]
                    
                    # 生成 Baseline 干净文本
                    with torch.no_grad():
                        clean_out = model.generate(input_ids, **gen_kwargs)
                        clean_ids = clean_out[0, input_ids.shape[1]:] if task == "generation" else clean_out[0, 1:]
                        clean_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
                    
                    # 生成 Watermark 文本 (加入 MLP Processor)
                    hsp_processor = ContextAwareHSPLogitsProcessor(mlp_net, llm_backbone, msg_tensor, embeddings_weight, alpha=CONFIG["alpha"])
                    gen_kwargs["logits_processor"] = LogitsProcessorList([hsp_processor])
                    
                    with torch.no_grad():
                        wm_out = model.generate(input_ids, **gen_kwargs)
                        wm_ids = wm_out[0, input_ids.shape[1]:] if task == "generation" else wm_out[0, 1:]
                        wm_text = tokenizer.decode(wm_ids, skip_special_tokens=True)
                        
                    # 计算 PPL 困惑度
                    clean_ppl_val, wm_ppl_val = 0.0, 0.0
                    if task == "generation":
                        clean_ppl_val = evaluator.calc_ppl(model, clean_out)
                        wm_ppl_val = evaluator.calc_ppl(model, wm_out)
                        
                    # 2. O(1) 盲提取原始文本的水印
                    extracted_msg = detector.extract_message(wm_text, capacity)
                    extracted_bits = [1 if x > 0 else 0 for x in extracted_msg.squeeze().tolist()]
                    bit_acc_clean = sum([1 for e, o in zip(extracted_bits, original_bits) if e == o]) / capacity
                    
                    # 3. 截断攻击 (Drop Attack) 及提取
                    crop_ids = perturb_attacker.attack_drop(wm_ids.tolist(), drop_ratio=0.3)
                    crop_text = tokenizer.decode(crop_ids, skip_special_tokens=True)
                    crop_extracted = detector.extract_message(crop_text, capacity)
                    crop_extracted_bits = [1 if x > 0 else 0 for x in crop_extracted.squeeze().tolist()]
                    bit_acc_crop = sum([1 for e, o in zip(crop_extracted_bits, original_bits) if e == o]) / capacity

                    # 4. 同义改写攻击 (Paraphrase Attack) 及提取
                    bit_acc_para = 0.0
                    para_text = ""
                    if para_attacker is not None:
                        para_text = para_attacker.attack(wm_text)
                        para_extracted = detector.extract_message(para_text, capacity)
                        para_extracted_bits = [1 if x > 0 else 0 for x in para_extracted.squeeze().tolist()]
                        bit_acc_para = sum([1 for e, o in zip(para_extracted_bits, original_bits) if e == o]) / capacity

                    actual_generated_tokens = len(wm_ids)
                    
                    record = {
                        "sample_id": idx + 1,
                        "prompt": prompt,
                        "original_bits": original_bits,
                        "clean_text": clean_text,
                        "watermarked_text": wm_text,
                        "paraphrased_text": para_text,
                        "generated_tokens": actual_generated_tokens,
                        "clean_ppl": round(clean_ppl_val, 4),
                        "wm_ppl": round(wm_ppl_val, 4),
                        "bit_acc_clean": bit_acc_clean,
                        "bit_acc_drop30": bit_acc_crop,
                        "bit_acc_paraphrase": bit_acc_para
                    }
                    all_results.append(record)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()

            valid_results = [r for r in all_results if r["generated_tokens"] >= filter_threshold]
            num_valid = len(valid_results)
            
            print(f"\n=== [全局指标汇总 Task: {task} | Cap: {capacity}b | 有效样本: {num_valid}/{len(all_results)}] ===")
            
            if num_valid == 0:
                print("    [!] 警告：没有达到长度阈值的样本，无法计算有效平均值。")
                continue
                
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
            
            res_str = f"    [平均比特准确率] 无攻击 Acc: {np.mean(bit_acc_clean_list)*100:.2f}% | 删减攻击 Acc: {np.mean(bit_acc_copy_list)*100:.2f}%"
            if para_attacker is not None:
                res_str += f" | 改写攻击 Acc: {np.mean(bit_acc_para_list)*100:.2f}%"
            print(res_str)
                
        del model
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context-Aware Multi-bit HSP Evaluation Engine")
    parser.add_argument("--task", type=str, default="generation", choices=["generation", "summarization", "translation", "all"])
    parser.add_argument("--paraphrase_model", type=str, default="none", choices=["none", "dipper", "pegasus"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--token_length", type=int, default=200)
    args = parser.parse_args()
    run_evaluation_pipeline(args)
