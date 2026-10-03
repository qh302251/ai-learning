import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_XET"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Tokenizer：把文字转成模型能理解的数字编号
model_path = "../models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# torch_dtype="auto" → 自动选最适合的精度（这个模型默认 float16）
# device_map="cuda:0"  → 把模型放到 GPU 上
save_path = "E:/models/qwen7b-4bit"

if os.path.exists(save_path):
    print("加载已保存的量化模型...")
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForCausalLM.from_pretrained(save_path, device_map="cuda:0")
else:
    print("正在加载模型（4-bit 量化），需要等几分钟...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cuda:0",
        quantization_config=bnb_config
    )
    print("保存量化模型到 E 盘，下次秒加载...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

prompt = "一艘船在河里航行，顺水每小时行20公里，逆水每小时行16公里。求船在静水中的速度和水流速度。"
messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": prompt}
]

# apply_chat_template：把 messages 组装成 Qwen 能理解的对话格式
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 真正转成数字编号，并放到模型所在的设备（GPU）上
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
# 把数字编号解码回文字
# 只取新生成的部分，去掉输入的对话模板
input_len = model_inputs["input_ids"].shape[1]
response = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
print("\n---模型回答---")
print(response)
