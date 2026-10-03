import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer,AutoModelForCausalLM

from peft import LoraConfig,get_peft_model
from trl import SFTTrainer
from trl import SFTConfig

base_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(base_dir,"..","models","models","Qwen--Qwen2.5-1.5B-Instruct","snapshots","master")
tokenizer=AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
train_data=load_from_disk(os.path.join(base_dir,"..","data_cleaned"))
print(f"数据集:{len(train_data)}条")

#loRA配置
lora_config=LoraConfig(
    r=8,    #秩（rank），控制LoRA学习能力，越大学得越多
    lora_alpha=16,   #缩放系数，控制LoRA影响强度
    target_modules=["q_proj","k_proj","v_proj","o_proj"], #插在Attention的Q/K/V/O上
    lora_dropout=0.05, #随机丢弃5%，防止过拟合
    bias="none",    #不训练偏置
    task_type="CAUSAL_LM"   #因果语言模型(逐词预测)
)

#把LoRA配置应用到模型上
model=get_peft_model(model,lora_config)
model.print_trainable_parameters()

#训练参数配置
training_args=SFTConfig(
    output_dir=os.path.join(base_dir,"..","output_1.5b"),#训练结果保存路径
    per_device_train_batch_size=2,#每张GPU一次处理两条数据
    gradient_accumulation_steps=4,#累积4次梯度再更新(等效batch_size=8)
    num_train_epochs=1,#完整训练1轮
    learning_rate=2e-4,#学习率
    fp16=True, #用float16训练，省显存
    save_steps=500,#每500步保存一次
    logging_steps=100,#每100步打印一次loss
    save_total_limit=2,#最多保留两个checkpoint
    remove_unused_columns=False, #保留数据集中的所有列
    report_to="none",#不向任何平台上报日志
    max_length=400,#最多截断到400个token，省显存
    )
def format_chat(example):
    text=f"用户:{example['instruction']}\n助手:{example['output']}"
    return text

#创建训练器
trainer=SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_data,
    formatting_func=format_chat,
    )
#开始训练
trainer.train()

#保存LoRA权重
trainer.save_model()