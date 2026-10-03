import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# ① 加载合并后的微调模型（LoRA 已融进权重，直接加载即可）
model_path = "E:/models/qwen15_merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# ② 创建一个 Flask 应用（这就是我们的 Web 服务）
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ②⑤ 访问根路径 / 时，返回聊天网页（打开浏览器就能聊）
@app.route("/")
def index():
    return open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8").read()


# 访问 /hao 时，返回清爽版"浩的 AI"页面
@app.route("/hao")
def index_hao():
    return open(os.path.join(BASE_DIR, "clean.html"), encoding="utf-8").read()


# ③ 定义接口：处理 POST /v1/chat/completions
#    路径故意和 OpenAI 官方 API 一致，将来想切换到别的模型都不用改客户端
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    # 取出客户端发来的 JSON 数据（标准 OpenAI 请求格式）
    data = request.get_json()
    messages = data.get("messages", [])  # 例：[{"role":"user","content":"你是谁"}]

    # 用聊天模板把对话拼成模型看得懂的格式
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("PROMPT:", repr(text))  # 调试：看看发给模型的内容到底长啥样
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 让模型生成回答
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
    )
    # 只取新生成的部分，转成文字
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 按 OpenAI 的响应格式返回（客户端就认这个格式）
    return jsonify({
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "model": "qwen15-finetuned",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
    })


if __name__ == "__main__":
    # 启动服务，监听 8000 端口（127.0.0.1 = 只允许本机访问）
    app.run(host="127.0.0.1", port=8000)
