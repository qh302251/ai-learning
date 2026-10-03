import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_XET"]="0"

from transformers import AutoTokenizer

model_path = "../models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
tokenizer=AutoTokenizer.from_pretrained(model_path)

#看看不同文本分别被拆成几个token
texts=[
    "今天天气真好",
    "今天",
    "天气",
    "真好",
    "今天的天",
    "今天的天真蓝",
]

for t in texts:
    ids=tokenizer(t)["input_ids"]
    tokens=[tokenizer.decode([id]) for id in ids]
    print(f"'{t}'->{len(ids)}个token:{tokens}")