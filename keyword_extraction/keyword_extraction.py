# 代码6–1
import jieba
import jieba.posseg
import numpy as np
import math
import operator
from gensim import corpora, models

# 代码6–2
# 获取停用词
def Stop_words():
    stopword = []
    data = []
    f = open('./data/stopword.txt', encoding='utf8')
    for line in f.readlines():
        data.append(line)
    for i in data:
        output = str(i).replace('\n', '')
        stopword.append(output)
    return stopword

# 采用jieba进行词性标注，对当前文档过滤词性和停用词
def Filter_word(text):
    filter_word = []
    stopword = Stop_words()
    text = jieba.posseg.cut(text)
    for word, flag in text:
        if flag.startswith('n') is False:
            continue
        if not word in stopword and len(word) > 1:
            filter_word.append(word)
    return filter_word
# 加载文档集，对文档集过滤词性和停用词
def Filter_words(data_path = './data/corpus.txt'):
    document = []
    for line in open(data_path, 'r', encoding='utf-8'):
        segment = jieba.posseg.cut(line.strip())
        filter_words = []
        stopword = Stop_words()
        for word, flag in segment:
            if flag.startswith('n') is False:
                continue
            if not word in stopword and len(word) > 1:
                filter_words.append(word)
        document.append(filter_words)
    return document

# 代码6–3
# TF-IDF 算法
def tf_idf():
    tf_idf_list = []
    # 统计TF值
    tf_dict = {}
    filter_word = Filter_word(text)
    for word in filter_word:
        if word not in tf_dict:
            tf_dict[word] = 1
        else:
            tf_dict[word] += 1
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / len(text)
    # 统计IDF值
    idf_dict = {}
    document = Filter_words()
    doc_total = len(document)
    for doc in document:
        for word in set(doc):
            if word not in idf_dict:
                idf_dict[word] = 1
            else:
                idf_dict[word] += 1
    for word in idf_dict:
        idf_dict[word] = math.log(doc_total / (idf_dict[word] + 1))
    # 计算TF-IDF值
    tf_idf_dict = {}
    for word in filter_word:
        if word not in idf_dict:
            idf_dict[word] = 0
        tf_idf_dict[word] = tf_dict[word] * idf_dict[word]
    # 提取前10个关键词
    keyword = 10
    print('TF-IDF模型结果：')
    for key, value in sorted(tf_idf_dict.items(), key=operator.itemgetter(1),
                             reverse=True)[:keyword]:
        print(key + '/', end='')
        tf_idf_list.append(key)
    print()
    print("tf_idf_list:",tf_idf_list)
    return tf_idf_list

# 代码6–4
def TextRank():
    TextRank_list = []
    window = 3
    win_dict = {}
    filter_word = Filter_word(text)
    length = len(filter_word)
    # 构建每个节点的窗口集合
    for word in filter_word:
        index = filter_word.index(word)
       # 设置窗口左、右边界，控制边界范围
        if word not in win_dict:
            left = index - window + 1
            right = index + window
            if left < 0:
                left = 0
            if right >= length:
                right = length
            words = set()
            for i in range(left, right):
                if i == index:
                    continue
                words.add(filter_word[i])
                win_dict[word] = words
    # 构建相连的边的关系矩阵
    word_dict = list(set(filter_word))
    lengths = len(set(filter_word))
    matrix = pd.DataFrame(np.zeros([lengths,lengths]))
    for word in win_dict:
        for value in win_dict[word]:
            index1 = word_dict.index(word)
            index2 = word_dict.index(value)
            matrix.iloc[index1, index2] = 1
            matrix.iloc[index2, index1] = 1
    summ = 0
    cols = matrix.shape[1]
    rows = matrix.shape[0]
    # 归一化矩阵
    for j in range(cols):
        for i in range(rows):
            summ += matrix.iloc[i, j]
        matrix[j] /= summ
    # 根据公式计算textrank值
    d = 0.85
    iter_num = 700
    word_textrank = {}
    textrank = np.ones([lengths, 1])
    for i in range(iter_num):
        textrank = (1 - d) + d * np.dot(matrix, textrank)
    # 将词语和textrank值一一对应
    for i in range(len(textrank)):
        word = word_dict[i]
        word_textrank[word] = textrank[i, 0]
    keyword = 10
    print('------------------------------')
    print('textrank模型结果：')
    for key, value in sorted(word_textrank.items(), key=operator.itemgetter(1),
                             reverse=True)[:keyword]:
        print(key + '/', end='')
        TextRank_list.append(key)
    print()
    print("TextRank_list:",TextRank_list)
    return TextRank_list

# 代码6–5
def lsi():
    lsi_list = []
    # 主题-词语
    document = Filter_words()
    dictionary = corpora.Dictionary(document)  # 生成基于文档集的语料
    corpus = [dictionary.doc2bow(doc) for doc in document]  # 文档向量化
    tf_idf_model = models.TfidfModel(corpus)  # 构建TF-IDF模型
    tf_idf_corpus = tf_idf_model[corpus]  # 生成文档向量
    lsi = models.LsiModel(tf_idf_corpus, id2word=dictionary, num_topics=4)
# 构建lsiLSI模型，函数包括3个参数：文档向量、文档集语料id2word和
# 主题数目num_topics，id2word可以将文档向量中的id转化为文字
    # 主题-词语
    words = []
    word_topic_dict = {}
    for doc in document:
        words.extend(doc)
        words = list(set(words))
    for word in words:
        word_corpus = tf_idf_model[dictionary.doc2bow([word])]
        word_topic= lsi[word_corpus]
        word_topic_dict[word] = word_topic
    # 文档-主题
    filter_word = Filter_word(text)
    corpus_word = dictionary.doc2bow(filter_word)
    text_corpus = tf_idf_model[corpus_word]
    text_topic = lsi[text_corpus]
    # 计算当前文档和每个词语的主题分布相似度
    sim_dic = {}
    for key, value in word_topic_dict.items():
        if key not in text:
            continue
        x = y = z = 0
        for tup1, tup2 in zip(value, text_topic):
            x += tup1[1] ** 2
            y += tup2[1] ** 2
            z += tup1[1] * tup2[1]
            if x == 0 or y == 0:
                sim_dic[key] = 0
            else:
                sim_dic[key] = z / (math.sqrt(x * y))
    keyword = 10
    print('------------------------------')
    print('LSI模型结果：')
    for key, value in sorted(sim_dic.items(), key=operator.itemgetter(1),
                            reverse=True)[: keyword]:
        print(key + '/' , end='')
        lsi_list.append(key)
    print()
    print("lsi_list:",lsi_list)
    return lsi_list

filename = r"D:\Users\IST\School\NLP\爬虫与关键词提取\tech_news.csv"
import pandas as pd
df = pd.read_csv(filename)

#指定分析第几条新闻
# num = 53
# print("原文:",df['内容'][num])
# text = df['内容'][num]
#
# keywords_list = df['关键词'][num].split(' ')
# print("keywords_list:",keywords_list,"len:",len(keywords_list))

for num in range(len(df)):
    def calculate_similarity(keywords_list, output_list):
        # 计算提取的关键词的字数
        extracted_keywords_length = 10
        # 计算新闻关键词的字数
        news_keywords_length = len(keywords_list)
        # 计算相同关键词的字数
        common_keywords = []

        common_keywords_length = len(
            [common_keywords.append(out_keyword) for out_keyword in output_list if out_keyword in keywords_list])

        # 计算相似度
        similarity = 2 * common_keywords_length / (extracted_keywords_length + news_keywords_length)

        print("common_keywords:", common_keywords)
        print(f"similarity = 2 *{common_keywords_length}/({extracted_keywords_length} + {news_keywords_length}) ")
        return similarity

    print("原文:", df['内容'][num])
    text = df['内容'][num]

    keywords_list = df['关键词'][num].split(' ')
    print("keywords_list:", keywords_list, "len:", len(keywords_list))

    tf_idf_list = tf_idf()
    tf_idf_similarity = calculate_similarity(keywords_list, tf_idf_list)
    print("tf_idf_相似度:", tf_idf_similarity)

    TextRank_list = TextRank()
    TextRank_similarity = calculate_similarity(keywords_list, TextRank_list)
    print("textrank_相似度:", TextRank_similarity)

    lsi_lisit = lsi()
    LSI_similarity = calculate_similarity(keywords_list, lsi_lisit)
    print("lsi_相似度:", LSI_similarity)
    print("\n")

# 代码6–6
#计算Top-K个词与新闻关键词的相似度，K=10，
#相似度S=2*相同关键词的字数/（提取的关键词字数+新闻关键词字数）

# def calculate_similarity(keywords_list, output_list):
#     # 计算提取的关键词的字数
#     extracted_keywords_length = 10
#     # 计算新闻关键词的字数
#     news_keywords_length = len(keywords_list)
#     # 计算相同关键词的字数
#     common_keywords = []
#
#     common_keywords_length = len([common_keywords.append(out_keyword) for out_keyword in output_list if out_keyword in keywords_list])
#
#     # 计算相似度
#     similarity = 2 * common_keywords_length / (extracted_keywords_length + news_keywords_length)
#
#     print("common_keywords:", common_keywords)
#     print(f"similarity = 2 *{common_keywords_length}/({extracted_keywords_length} + {news_keywords_length}) ")
#     return similarity


# tf_idf_list = tf_idf()
# tf_idf_similarity =  calculate_similarity(keywords_list, tf_idf_list)
# print("tf_idf_相似度:", tf_idf_similarity)
#
# TextRank_list = TextRank()
# TextRank_similarity =  calculate_similarity(keywords_list, TextRank_list)
# print("textrank_相似度:", TextRank_similarity)
#
# lsi_lisit = lsi()
# LSI_similarity = calculate_similarity(keywords_list,lsi_lisit)
# print("textrank_相似度:", LSI_similarity)