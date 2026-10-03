"""构建知识库的向量索引（Embedding），供语义搜索使用"""

import os
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from sentence_transformers import SentenceTransformer
from tools.chunk_utils import chunk_text_v2

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")


def main():
    txt_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".txt")]
    if not txt_files:
        print(f"知识库目录为空: {KNOWLEDGE_DIR}")
        print(f"请先放一些 .txt 文档到 {KNOWLEDGE_DIR}")
        return

    all_chunks = []

    for f in txt_files:
        filepath = os.path.join(KNOWLEDGE_DIR, f)
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()
        chunks = chunk_text_v2(content, f)
        all_chunks.extend(chunks)
        print(f"  读取: {f}")

    print(f"加载语义搜索模型...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = [c["text"] for c in all_chunks]
    print(f"编码 {len(texts)} 个文本块...")
    embeddings = model.encode(texts).tolist()

    output = {
        "chunks": all_chunks,
        "embeddings": embeddings,
    }
    out_path = os.path.join(KNOWLEDGE_DIR, "embeddings.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"索引已保存: {out_path}")


if __name__ == "__main__":
    main()



