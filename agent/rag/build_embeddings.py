"""
把 knowledge/ 里的文档转成向量，保存到文件
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from sentence_transformers import SentenceTransformer

# 加载 embedding 模型（自动下载，第一次需要联网）
print("正在加载模型...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("模型加载完成")

# 读取 knowledge 目录下所有 .txt 文件
docs = []
filenames = []

for f in os.listdir(os.path.join(os.path.dirname(__file__), "knowledge")):
    if f.endswith(".txt"):
        filepath = os.path.join(os.path.dirname(__file__), "knowledge", f)
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()
        docs.append(content)
        filenames.append(f)

print(f"共读取 {len(docs)} 个文档")

# 转成向量
embeddings = model.encode(docs)
print(f"向量维度: {embeddings.shape}")  # 例如 (3, 384) 表示3个文档，每个384维

# 保存到文件
data = {
    "filenames": filenames,
    "documents": docs,
    "embeddings": embeddings.tolist(),  # numpy 转 list 才能 JSON 序列化
}
with open(os.path.join(os.path.dirname(__file__), "knowledge", "embeddings.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

print("向量已保存到 knowledge/embeddings.json")
