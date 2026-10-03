import os
import sys
import tempfile
import unittest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.rag_tools import search_documents, hybrid_search
from tools.bm25_tools import build_stats


class TestRagTools(unittest.TestCase):
    def setUp(self):
        """每个测试前创建临时知识库目录"""
        self.knowledge_dir = tempfile.mkdtemp()

    # ==================== 关键词搜索 ====================

    def test_search_document_found(self):
        """搜索关键词存在的文档"""
        path = os.path.join(self.knowledge_dir, "test.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("机器学习是人工智能的一个分支")
        result = search_documents("机器学习", self.knowledge_dir)
        self.assertIn("test.txt", result)
        self.assertIn("机器学习", result)

    def test_search_document_not_found(self):
        """搜索不存在的关键词，返回提示信息"""
        path = os.path.join(self.knowledge_dir, "test.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("机器学习是人工智能的一个分支")
        result = search_documents("深度学习", self.knowledge_dir)
        self.assertIn("未找到", result)

    # ==================== 混合检索（BM25 + 向量 + RRF）====================
    # 用 mock 把模型/向量/索引全部替换成假对象，不加载任何真模型即可测检索逻辑

    def test_hybrid_search_not_ready(self):
        """模型或索引未就绪时，返回提示信息而不是抛异常"""
        with patch("tools.rag_tools._model", None), \
             patch("tools.rag_tools._doc_embeddings", None):
            result = hybrid_search("测试")
        self.assertIn("未就绪", result)

    @patch("tools.rag_tools._model")
    @patch("tools.rag_tools._doc_embeddings", new=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    @patch("tools.rag_tools._bm25_docs", new=[["苹果", "水果"], ["太阳", "升起"]])
    @patch("tools.rag_tools._bm25_stats", new=build_stats([["苹果", "水果"], ["太阳", "升起"]]))
    @patch("tools.rag_tools._reranker", new=None)  # 走"无 reranker"的兜底路径
    @patch(
        "tools.rag_tools._emb_data",
        new={
            "chunks": [
                {"text": "苹果是一种水果", "source": "fruit.txt", "chunk_index": 0},
                {"text": "太阳从东边升起", "source": "sun.txt", "chunk_index": 0},
            ]
        },
    )
    def test_hybrid_search_ready(self, mock_model):
        """模型就绪时，混合检索返回最匹配的切片"""
        mock_model.encode.return_value = np.array([0.9, 0.1, 0.0])  # 与第 0 条文档更相似
        result = hybrid_search("水果", top_n=1)
        self.assertIn("苹果", result)       # 召回了正确切片
        self.assertNotIn("太阳", result)    # 且只返回 top_n=1 个

    def tearDown(self):
        import shutil
        shutil.rmtree(self.knowledge_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()