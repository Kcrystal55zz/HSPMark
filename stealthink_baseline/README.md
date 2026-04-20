# StealthInk Baseline

This directory contains the **StealthInk** watermarking method for LLMs, added as a baseline for comparison experiments.

**Source:** [yajiang4215/StealthInk_A-Multi-bit-and-Stealthy-Watermark-for-Large-Language-Models](https://github.com/yajiang4215/StealthInk_A-Multi-bit-and-Stealthy-Watermark-for-Large-Language-Models)

## Files

| File | Description |
|------|-------------|
| `every_step_1_24_direct_detect.py` | Main script: watermark generation and detection pipeline |
| `every_step_processor.py` | `WatermarkBase` class with RNG seeding scheme |
| `hash_scheme.py` | PRF (pseudo-random function) implementations and seeding scheme lookup |
| `normalizers.py` | Text normalizers (Unicode, homoglyphs, truecase) |

## Usage

Run the main script from **inside** the `stealthink_baseline/` directory (so that relative imports resolve correctly):

```bash
cd stealthink_baseline
python3 every_step_1_24_direct_detect.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --chunk_capacity 1 \
    --msg_len 24 \
    --generation_num 500 \
    --generation_length 300 \
    --prompts_fp c4_subset_10000_15_20words_prompts.json \
    --out_dir output
```

Or run from the repo root by adding the directory to `PYTHONPATH`:

```bash
PYTHONPATH=stealthink_baseline python3 stealthink_baseline/every_step_1_24_direct_detect.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --chunk_capacity 1 \
    --msg_len 24 \
    --generation_num 500 \
    --generation_length 300 \
    --prompts_fp stealthink_baseline/c4_subset_10000_15_20words_prompts.json \
    --out_dir output
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name_or_path` | `meta-llama/Llama-2-7b-hf` | HuggingFace model ID or local path |
| `--chunk_capacity` | `1` | Bits embedded per token (chunk) |
| `--msg_len` | `24` | Total bits in the embedded message |
| `--generation_num` | `500` | Number of prompts to process |
| `--generation_length` | `300` | Number of new tokens per response |
| `--prompts_fp` | `c4_subset_10000_15_20words_prompts.json` | Path to the prompts JSON file |
| `--out_dir` | `output` | Directory to write results |
| `--sampling_temp` | `1.0` | Sampling temperature |
| `--use_sampling` | `True` | Use multinomial sampling |
| `--load_fp16` | `False` | Load model in float16 |

## Dependencies

Install required packages (in addition to those in the main `requirements.txt`):

```bash
pip install homoglyphs scikit-learn tokenizers
```
