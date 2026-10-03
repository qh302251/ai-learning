import os 
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

base_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(base_dir,"..","models","models","Qwen--Qwen2.5-1.5B-Instruct","snapshots","master")
lora_path=os.path.join(base_dir,"..","output_1.5b")

tokenizer=AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model=PeftModel.from_pretrained(model,lora_path)

messages=[
    {"role":"user","content":"请用鲁迅风格写一句鼓励的话"}
]
text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
print("用户:",messages[0]["content"])

inputs=tokenizer(text,return_tensors="pt").to("cuda")
outputs=model.generate(**inputs,max_new_tokens=200)
response=tokenizer.decode(outputs[0],skip_special_tokens=True)
print(response)

