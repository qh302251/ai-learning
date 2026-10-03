import re 
import jieba

def tokenize_jieba(text):
    tokens=[]
    for w in jieba.cut(text.lower()):
        if re.search(r'\w',w):
            tokens.append(w)
    return tokens

if __name__=="__main__":
    print(tokenize_jieba("神经网络用于图像识别,Transformer是核心架构。"))