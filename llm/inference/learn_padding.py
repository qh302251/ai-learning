import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_XET"]="0"

from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("../models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master")

long_text = ["你好", "今天天气真好","我非常喜欢学习人工智能相关的知识因为它真的很有趣"]
# batch = tokenizer(long_text, padding="max_length", max_length=8,truncation=True)
batch = tokenizer(long_text, padding="max_length", max_length=8)
print("input_ids:", batch["input_ids"])
print("attention_mask:", batch["attention_mask"])
print("截断后:", tokenizer.decode(batch["input_ids"][2]))

print("BOS:", tokenizer.bos_token, "→", tokenizer.bos_token_id)
print("EOS:", tokenizer.eos_token, "→", tokenizer.eos_token_id)
print("PAD:", tokenizer.pad_token, "→", tokenizer.pad_token_id)