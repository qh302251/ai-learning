import os 
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

base_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(base_dir,"..","models","models","Qwen--Qwen2.5-1.5B-Instruct","snapshots","master")
lora_path=os.path.join(base_dir,"..","output_1.5b")
save_path="E:/models/qwen15_merged"

#加载tokenizer和原始模型
tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(
model_path,
device_map="auto"
)
#挂上LoRA
model=PeftModel.from_pretrained(model,lora_path)

#合并LoRA进原始模型
merged_model=model.merge_and_unload()
#保存合并后的完整模型
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
