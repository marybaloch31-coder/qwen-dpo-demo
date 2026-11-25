import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ====== CHANGE THIS to your Hugging Face model repo ======
ADAPTER_REPO = "YOUR_USERNAME/qwen2.5-0.5b-dpo-lab10"
# =========================================================
BASE_MODEL = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

_cache = {}

def get_model(choice):
    if choice in _cache:
        return _cache[choice]
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    if choice == "Base Model":
        model = base
    else:
        subfolder = "beta-0.1" if "0.1" in choice else "beta-0.5"
        model = PeftModel.from_pretrained(base, ADAPTER_REPO, subfolder=subfolder)
    model.eval()
    _cache[choice] = model
    return model

def generate(prompt, choice, max_new_tokens):
    model = get_model(choice)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=4, label="Prompt"),
        gr.Radio(["Base Model", "DPO Beta 0.1", "DPO Beta 0.5"], value="DPO Beta 0.1", label="Model"),
        gr.Slider(30, 200, value=120, step=10, label="Max new tokens"),
    ],
    outputs=gr.Textbox(lines=10, label="Answer"),
    title="Qwen2.5-0.5B DPO Alignment (Lab 10)",
    description="Compare the base model with DPO-aligned versions (beta 0.1 and 0.5).",
    examples=[
        ["Explain how Deep Learning works in simple language with a real-life example.", "DPO Beta 0.1", 120],
        ["Give five useful tips for a university student who wants to improve time management.", "DPO Beta 0.5", 120],
    ],
)

if __name__ == "__main__":
    demo.launch()
