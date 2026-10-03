import os
from datasets import load_from_disk
from transformers import AutoTokenizer
base_dir = os.path.dirname(os.path.abspath(__file__))
train_data = load_from_disk(os.path.join(base_dir, "..", "data_cleaned"))

# 打印第一条数据
item = train_data[0]
print("字段:", list(item.keys()))
print("=" * 40)
print("指令:", item["instruction"])
print("-" * 40)
print("输出:", item["output"])

#模拟format_chat拼接后的样子
text=f"用户:{item['instruction']}\n助手:{item['output']}"
print("=*40")
print("format_chat拼接后: ")
print(text)

tokenizer=AutoTokenizer.from_pretrained(os.path.join(base_dir,"..","models","models","Qwen--Qwen2.5-1.5B-Instruct","snapshots","master"))
messages=[{"role":"user","content":item["instruction"]}]
print("="*40)
print("套模版后:")
print(tokenizer.apply_chat_template(messages,tokenize=False))