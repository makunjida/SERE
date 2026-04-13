#!/usr/bin/env python
# -*- coding:utf-8 -*-
#分析文本数据中的句法结构
import os
import pickle
import string
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
from tqdm import tqdm

nlp = StanfordCoreNLP('../../stanford-corenlp-4.5.6', lang='en')
#nlp = StanfordCoreNLP('D:/ProgramData/stanford-corenlp-full-2016-10-31', lang='en')


#路径设置
os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.path.abspath(os.path.dirname(os.getcwd()))
os.path.abspath(os.path.join(os.getcwd(), ".."))
##修改文件名qprop_small为qprop
dataset ='qprop'

input = os.sep.join(['..','..', 'data_tgcn', dataset, dataset+"_new"])
output = os.sep.join(['..','..', 'data_tgcn', dataset, 'stanford'])
if not os.path.exists(output):
    os.mkdir(output)


#读取病例
yic_content_list = []
f = open(input + '.clean.txt', 'r', encoding="gbk")
lines = f.readlines()
for line in lines:
    yic_content_list.append(line.strip())#文本条
f.close()

import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#获取句法依存关系对
rela_pair_count_str = {}
for doc_id in tqdm(range(0,len(yic_content_list2))): #使用 tqdm 进度条包装器遍历文档列表，提供一个用户友好的进度条。
# for doc_id in range(len(yic_content_list2)):
#     print(doc_id)
    words = yic_content_list2[doc_id]
    words = words.split("\n")
    rela=[]
    for window in words:
        if window==' ':
            continue
        #构造rela_pair_count
        window = window.replace(string.punctuation, ' ')#把所有的标点符号替换为' '
        ws=window.split(" ")
        res = nlp.dependency_parse(window) #生成句法依存对   res 包含依存关系的类型和涉及的词索引。
        distances = calculate_distance(res, len(ws))
        for tuple in res:
            # rela.append(str(tuple[0]) + ', ' + str(tuple[1]))
            rela.append(str(tuple[0]) + ', ' + str(tuple[1])+', '+str(tuple[2])) #rela.append(...): 将解析得到的依存关系（类型和词索引）格式化为字符串并添加到列表 rela 中。
            
        for pair in rela:
            #print(pair)
            pair=pair.split(", ")
            rel_type = pair[0]  # 关系类型（未用于计算，但可用于其他目的）
            gov_index = int(pair[1])  # 主控词索引
            dep_index = int(pair[2])  # 从属词索引

            # 计算索引的绝对差值
            index_difference = abs(gov_index - dep_index)
            if pair[0]=='ROOT' or pair[1]=='ROOT':
                continue
            # if pair[0] == pair[1]:
            if pair[2] == pair[1]:
                continue
            if int(pair[1])>len(ws) or int(pair[2])>len(ws):
                continue
            # if pair[0] in string.punctuation or pair[1] in string.punctuation:
            if ws[int(pair[2])-1] in string.punctuation or ws[int(pair[1])-1] in string.punctuation:
                continue
            # if pair[0] in stop_words or pair[1] in stop_words:
            if ws[int(pair[1])-1] in stop_words or ws[int(pair[2])-1] in stop_words:
                continue
            # word_pair_str = pair[0]+ ',' + pair[1]
            word_pair_str = ws[int(pair[1])-1] + ',' + ws[int(pair[2])-1]
            #print(index_difference)
            #print(pair)
            #print(word_pair_str)   #word_pair_str包含的内容为单词对
            if word_pair_str in rela_pair_count_str and index_difference < 4 :
                rela_pair_count_str[word_pair_str] += 1
            else:
                if word_pair_str not in rela_pair_count_str and index_difference <= 4:
                    rela_pair_count_str[word_pair_str] = 1  # 将该键的值设为1
            # two orders
            # word_pair_str = pair[1]+ ',' + pair[0]
            # print(int(pair[2])-1,int(pair[1])-1)
            word_pair_str = ws[int(pair[2])-1] + ',' + ws[int(pair[1])-1]
            if word_pair_str in rela_pair_count_str and index_difference < 3:
                rela_pair_count_str[word_pair_str] += 1
            else:
                if word_pair_str not in rela_pair_count_str and index_difference <= 4:
                    rela_pair_count_str[word_pair_str] = 1  # 将该键的值设为1
            #print(rela_pair_count_str[word_pair_str])
            
#print(rela_pair_count_str)

#将rela_pair_count_str存成pkl格式
output1=open(output + '/{}_stan.pkl'.format(dataset),'wb')
pickle.dump(rela_pair_count_str, output1)