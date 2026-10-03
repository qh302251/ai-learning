import re

"""文本分块 — 将长文档切成固定大小的段落"""
def chunk_text(text: str, source: str, chunk_size: int = 300, overlap: int = 30) -> list[dict]:
    """将文本切成固定大小的块
    Args:
        text: 原始文本
        source: 来源文件名
        chunk_size: 每块最大字符数
        overlap: 块间重叠字符数
    Returns:
        list[dict]: [{"text": "...", "source": "...", "chunk_index": 0}, ...]
    """
    if len(text) <= chunk_size:
        return [{"text": text, "source": source, "chunk_index": 0}]

    chunks = []
    start = 0
    i = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({"text": chunk, "source": source, "chunk_index": i})
        start += chunk_size - overlap
        i += 1

    return chunks 


def chunk_text_v2(text: str, source: str, target_size: int = 400, min_size: int = 150) -> list[dict]:
    """句子级切分 + 累积合并（PDF 换行不可靠，跳过段落级）"""
    # 1 预处理：清掉 PDF 提取噪声
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]", "", text)

    # 2 切句子
    sents = [s.strip() for s in re.split(r"(?<=[.?!。？！])", text) if s.strip()]   

    # 3 兜底：超长的"句子"其实是一坨没有句号的文本（标题+作者+单位），按换行再切
    fixed = []
    for s in sents:
        if len(s) >= target_size:                        
            fixed.extend(x.strip() for x in s.split("\n") if x.strip())          
        else:
            fixed.append(s)
    sents = fixed

    # 4 累积合并：攒到接近 target_size 才封块
    chunks = []
    buf = ""
    for sent in sents:
        if len(buf + sent) >= target_size and len(buf) >= min_size:            # ← 空2
            chunks.append(buf)
            buf = ""
        buf += sent
    if buf:
        chunks.append(buf)

    return [{"text": c, "source": source, "chunk_index": i} for i, c in enumerate(chunks)]

if __name__ == "__main__":
    txt = open(r"H:\ai-learning\agent\project\assistant\knowledge\long.txt", encoding="utf-8").read()
    cs = chunk_text_v2(txt, "long.txt")
    print(f"共 {len(cs)} 块")
    for c in cs[:6]:
        print(f"--- idx={c['chunk_index']} len={len(c['text'])}")
        print("  开头:", repr(c["text"][:50]))
        print("  结尾:", repr(c["text"][-50:]))