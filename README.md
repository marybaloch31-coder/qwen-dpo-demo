---
title: Qwen DPO Alignment Lab 10
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Qwen2.5-0.5B DPO Alignment (Lab 10)

This project aligns **Qwen/Qwen2.5-0.5B** using **Direct Preference Optimization (DPO)**
with LoRA adapters.

## Training setup
- Base model: `Qwen/Qwen2.5-0.5B`
- Dataset: `trl-lib/ultrafeedback_binarized` (1,000 train / 50 eval examples)
- Method: DPO with LoRA (r=8, alpha=16, dropout=0.05, all attention + MLP projections)
- Learning rate: 5e-6, 100 steps, batch size 1, grad accumulation 4, fp16
- Two experiments: **beta = 0.1** and **beta = 0.5**

## Results
Both models and the base model were tested on 6 prompts (explanation, advice,
summarisation, professional writing, reasoning) and rated 1–5 on Helpfulness,
Relevance, Clarity and Instruction Following. See the dataset repo for the CSVs.

## Links
- Model: https://huggingface.co/YOUR_USERNAME/qwen2.5-0.5b-dpo-lab10
- Dataset: https://huggingface.co/datasets/YOUR_USERNAME/dpo-lab10-results
- Demo: https://huggingface.co/spaces/YOUR_USERNAME/qwen-dpo-demo

## Files
- `app.py` – Gradio demo
- `requirements.txt` – dependencies
