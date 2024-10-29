import spacy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('vader_lexicon')

# 加载 spacy 模型
nlp = spacy.load('zh_core_web_sm')  # 使用中文的 Spacy 模型

# 初始化 VADER 情感分析器
sid = SentimentIntensityAnalyzer()

# 示例文本处理
folder_path = '/home/xuezhi/Desktop/rawcn'  # .txt 文件的路径

# 定义要分析的方面
aspects = ['参与', '隐私', '访问', '教学', '学习']  # 中文关键词

# 用于存储结果的列表
results = []

# 处理文件夹中的每个文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # 仅处理 .txt 文件
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # 使用 Spacy 处理中文文本
        doc = nlp(text)
        
        # 字典用于存储每个方面的情感分数
        aspect_sentiment = {aspect: [] for aspect in aspects}
        
        # 提取句子并对每个方面进行情感分析
        for sent in doc.sents:
            for aspect in aspects:
                if aspect in sent.text.lower():  # 检查句子中是否包含中文关键词
                    sentiment = sid.polarity_scores(sent.text)
                    # 存储综合情感分数（整体情感）
                    aspect_sentiment[aspect].append(sentiment['compound'])
        
        # 计算每个方面的整体情感（综合分数的平均值）
        overall_sentiment = {aspect: np.mean(scores) if scores else 0 for aspect, scores in aspect_sentiment.items()}
        
        # 将该文件的结果添加到列表中
        result = {'file': filename.replace('.txt', '')}
        result.update(overall_sentiment)
        results.append(result)

# 将结果转换为 DataFrame
df = pd.DataFrame(results)

# 将结果保存为 CSV 文件
output_csv = '/home/xuezhi/Desktop/aspect_sentiment_analysis_resultscn.csv'
df.to_csv(output_csv, index=False)

print(f"结果已保存到 {output_csv}")

