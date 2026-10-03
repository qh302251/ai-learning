import torch

print(f"Pytorch版本:{torch.__version__}")
print(f"CUDA是否可用:{torch.cuda.is_available()}")
print(f"GPU名称:{torch.cuda.get_device_name(0)}")
total_vram=torch.cuda.get_device_properties(0).total_memory /1024**3
print(f"总显存:{total_vram:.2f} GB")

def test_tensor_size(name,num_params,dtype):
    try:
        tensor=torch.zeros(num_params,dtype=dtype,device="cuda")
        mem=torch.cuda.memory_allocated() /1024**3
        print(f"{name}:{mem:.3f} GB")
        del tensor
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"{name}:OOM!({e})")

print("\n=== float32 测试（4字节/参数）===")
sizes=[0.5,1,2,3]
for b in sizes:
    test_tensor_size(f"{b}B参数",int(b*1e9),torch.float32)

print("\n=== float16 测试（2字节/参数）===")
for b in sizes:
    test_tensor_size(f"{b}B 参数", int(b * 1e9), torch.float16)

print("\n=== float16 测试（1字节/参数）===")
for b in sizes:
    test_tensor_size(f"{b}B 参数", int(b * 1e9), torch.int8)

print("\n=== 总结 ===")
print(f"总显存: {total_vram:.2f} GB")
print("模型大小估算（粗略）：")
print(f"  7B 模型 float32: {7 * 4:.1f} GB → 跑不动")
print(f"  7B 模型 float16: {7 * 2:.1f} GB → 极限")
print(f"  7B 模型 4-bit:  ~{7 * 0.5:.1f} GB → 可行!")
print(f"  3B 模型 float16: {3 * 2:.1f} GB → 轻松跑")