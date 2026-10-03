import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_XET"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = """三个人三天喝三桶水，九个人九天喝几桶水？
请一步步推理，最后给出答案。注意：这个问题有陷阱。"""

messages = [
    {"role": "system", "content": "你是一个有帮助的助手。请一步步推理。"},
    {"role": "user", "content": prompt}
]

# ---------- 1.5B ----------
print("=" * 60)
print("【Qwen2.5-1.5B】")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", cache_dir="../models")
model_15 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", cache_dir="../models", torch_dtype="auto", device_map="cuda:0"
)

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model_15.device)
out = model_15.generate(**inputs, max_new_tokens=512, do_sample=False)
resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(resp)

del model_15
import torch; torch.cuda.empty_cache()

# ---------- 7B ----------
print("\n" + "=" * 60)
print("【Qwen2.5-7B (4-bit)】")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("E:/models/qwen7b-4bit")
model_7 = AutoModelForCausalLM.from_pretrained("E:/models/qwen7b-4bit", device_map="cuda:0")

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model_7.device)
out = model_7.generate(**inputs, max_new_tokens=512, do_sample=False)
resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(resp)
