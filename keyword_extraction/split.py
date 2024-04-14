"""

#代码4–2
import jieba
text = '中文分词是自然语言处理的一部分！'
seg_list = jieba.cut(text, cut_all=True)
print('全模式：', '/ ' .join(seg_list))
seg_list = jieba.cut(text, cut_all=False)
print('精确模式：', '/ '.join(seg_list))
seg_list = jieba.cut_for_search(text)
print('搜索引擎模式', '/ '.join(seg_list))

"""
import jieba
import csv

filename = r"D:\Users\IST\School\NLP\爬虫与关键词提取\tech_news.csv"

import pandas as pd
df = pd.read_csv(filename)
print(df['内容'][1])

text = df['内容'][1]
seg_list = jieba.cut(text, cut_all=True)
print('全模式：', '/ ' .join(seg_list))

