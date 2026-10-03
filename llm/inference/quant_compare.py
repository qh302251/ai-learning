import os 
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_XET"]="0"

import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

model_path="../models/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(model_path,
                                           torch_dtype="auto",
                                           device_map="cuda:0")

# 看模型第一层的权重
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, dtype={param.dtype}")
    print(f"  前 10 个值: {param.flatten()[:10]}")
    break  # 只看第一层

original_weight = model.model.embed_tokens.weight.data
print(f"原始权重: {original_weight[:5, :5]}")

# 模拟 4-bit 量化：先找出这组权重的范围，然后压缩到 16 个等级
w = original_weight[:100, :100].flatten()  # 取一小部分做演示
w_min, w_max = w.min(), w.max()
scale = (w_max - w_min) / 15  # 4-bit = 16 个等级

# 量化：把实际值映射到 0~15 的整数
quantized = ((w - w_min) / scale).round().clamp(0, 15)
# 反量化：从整数还原成近似值
dequantized = quantized * scale + w_min

print(f"\n原始值范围: {w_min:.6f} ~ {w_max:.6f}")
print(f"量化后 (前20个整数): {quantized[:20].int().tolist()}")
print(f"反量化后 (前20个): {dequantized[:20].tolist()}")
print(f"\n对比前 10 个值:")
for i in range(10):
    print(f"原始 {w[i].item():.6f} → 量化 {int(quantized[i].item())} →还原 {dequantized[i].item():.6f}")