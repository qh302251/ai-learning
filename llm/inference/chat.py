import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_XET"]="0"
from transformers import AutoTokenizer,AutoModelForCausalLM

model_path="../models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"

print("加载模型中...")
tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(
    model_path,torch_dtype="auto",device_map="cuda:0"
)
print("加载完成！输入 exit 退出聊天。\n")

history=[{"role":"system","content":"你是一个有用的助手。"}]

while True:
    user_input=input("\n你:")
    if user_input.lower() in ["exit","quit"]:
        break
    history.append({"role":"user","content":user_input})
    
    text=tokenizer.apply_chat_template(history,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer([text],return_tensors="pt").to(model.device)

    output=model.generate(
        **inputs,max_new_tokens=512,do_sample=True,temperature=0.7,
        top_p=0.9
    )
    response=tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
    
    print(f"AI:{response}")
    history.append({"role":"assistant","content":response})               