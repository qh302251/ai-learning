# -*- coding: utf-8 -*-
import re
import jieba

print("=== 实验1: jieba.cut 到底返回什么 ===")
s = "神经网络用于图像识别"
print("原始字符串:", s)
print("jieba.cut(s) 直接打印:", jieba.cut(s))
print("用 list() 包起来:", list(jieba.cut(s)))

print()
print("=== 实验2: 分词结果里混了哪些脏东西 ===")
s2 = "神经网络用于图像识别，Transformer 是核心架构。"
for w in jieba.cut(s2):
    print("  token =", repr(w))

print()
print("=== 实验3: re 是什么 —— 用模式去匹配字符串 ===")
print("re.search(r'\\w', '神经网络') ->", re.search(r"\w", "神经网络"))
print("re.search(r'\\w', '，')     ->", re.search(r"\w", "，"))
print("re.search(r'\\w', ' ')      ->", re.search(r"\w", " "))
print()
print("re.fullmatch(r'[\\W_]+', '，') ->", re.fullmatch(r"[\W_]+", "，"))
print("re.fullmatch(r'[\\W_]+', '的') ->", re.fullmatch(r"[\W_]+", "的"))

print()
print("=== 实验4: 两种判断方式，逐个词过一遍 ===")
for w in ["神经网络", "，", " ", "Transformer", "。", "的"]:
    keep_search = bool(re.search(r"\w", w))
    is_punct    = bool(re.fullmatch(r"[\W_]+", w))
    print(f"  {repr(w):16} search找到字母/汉字={keep_search!s:5} fullmatch是纯标点={is_punct}")