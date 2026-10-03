import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 合并后的完整模型（LoRA 已融进去，不需要 PeftModel 再挂）
model_path = "E:/models/qwen15_merged"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 用聊天模板格式化问题（顺便验证模板有没有丢）
messages = [{"role": "user", "content": "你是谁"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("发送给模型的内容：", text)

# 转成模型输入
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 生成回答
output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
)

# 只取新生成的部分（去掉输入）
answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("模型的回答：", answer)
