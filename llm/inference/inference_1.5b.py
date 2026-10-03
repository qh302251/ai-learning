import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_XET"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "../models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"

print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("正在加载 1.5B 模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="cuda:0"
)

prompt = "一艘船在河里航行，顺水每小时行20公里，逆水每小时行16公里。求船在静水中的速度和水流速度。"
messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("正在生成回复...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2
)
input_len = model_inputs["input_ids"].shape[1]
response = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
print("\n---1.5B 回答---")
print(response)
