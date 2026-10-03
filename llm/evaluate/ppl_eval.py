#Perplexity(困惑度)
import os 
import math
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

base_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(base_dir,"..","models","models","Qwen--Qwen2.5-1.5B-Instruct","snapshots","master")
lora_path=os.path.join(base_dir,"..","output_1.5b")
def compute_ppl(model,tokenize,text):
    #1.把文字转成token ids，放到GPU
    inputs=tokenize(text,return_tensors="pt").to("cuda")

    #2.不计算梯度(评估不需要训练)
    with torch.no_grad():
        outputs=model(**inputs,labels=inputs["input_ids"])

    #3.loss就是模型对这段文字的交叉熵损失
    loss=outputs.loss.item()

    #4.困惑度=e^loss
    ppl=math.exp(loss)
    return ppl

# 测试文本：自己写的，模型没训练过的内容（覆盖不同主题）
test_texts = [
    "中国航天员在空间站完成了多项科学实验，包括微重力条件下的材料制备和细胞培养研究，为未来深空探索积累了宝贵数据。",
    "昨晚我去楼下新开的餐厅吃饭，点了他们家的招牌红烧肉，味道很不错，价格也实惠，下次还想带朋友一起去。",
    "秋天到了，院子里的银杏树叶子开始变黄，一阵风吹过，落叶像蝴蝶一样飘落下来，铺满了整条小径。",
    "学习编程最重要的是动手实践，光看书很难真正理解，只有自己写代码、调试错误，才能慢慢掌握其中的原理。",
    "这个周末我打算去爬山，顺便带上相机拍些风景照，回来后挑几张满意的发到朋友圈，和朋友们分享一下。",
]

tokenizer=AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token=tokenizer.eos_token

def calc_average_ppl(model):
    ppls=[]
    for text in test_texts:
        ppls.append(compute_ppl(model,tokenizer,text))
    avg=sum(ppls)/len(ppls)
    return avg

# ========== 原始模型 ==========
print("计算原始模型ppl")
model=AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"    
)
avg_ppl=calc_average_ppl(model)
print(f"原始模型平均PPL:{avg_ppl:.2f}")
del model
torch.cuda.empty_cache()

# ========== 微调后模型 ==========
print("计算微调后模型PPL...")
model=AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model=PeftModel.from_pretrained(model,lora_path)
avg_ppl=calc_average_ppl(model)
print(f"微调后模型评价PPL:{avg_ppl:.2f}")