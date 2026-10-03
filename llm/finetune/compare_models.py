import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "models", "models",
                          "Qwen--Qwen2.5-1.5B-Instruct", "snapshots", "master")

lora_4500 = os.path.join(base_dir, "..", "output_1.5b","checkpoint-4500")
lora_4809 = os.path.join(base_dir, "..", "output_1.5b","checkpoint-4809")
# lora_path = os.path.join(base_dir, "..", "output_1.5b")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# 测试问题
questions = [
    "请用鲁迅风格写一句鼓励的话",
    "什么是机器学习？请用通俗的语言解释",
    "如何制定一个有效的学习计划？",
]

def generate(model,question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # ========== 原始模型（不挂 LoRA） ==========
# print("=" * 30)
# print("【原始模型】")
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# for q in questions:
#     print(f"\nQ: {q}")
#     print(generate(model, q))
# del model
# torch.cuda.empty_cache()


# # ========== 微调模型（挂 LoRA） ==========
# print("=" * 30)
# print("【微调后模型】")
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# model = PeftModel.from_pretrained(model, lora_path)
# for q in questions:
#     print(f"\nQ: {q}")
#     print(generate(model, q))

def test_lora(lora_path, label):
    print("=" * 40)
    print(f"【{label}】")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_path)
    for q in questions:
        print(f"\nQ: {q}")
        print(generate(model, q))
    del model
    torch.cuda.empty_cache()


test_lora(lora_4500, "checkpoint-4500")
test_lora(lora_4809, "checkpoint-4809")