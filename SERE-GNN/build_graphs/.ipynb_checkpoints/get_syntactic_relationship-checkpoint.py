#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pickle
import string
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
from tqdm import tqdm


nlp = StanfordCoreNLP('D:/ProgramData/stanford-corenlp-full-2016-10-31', lang='en')


#路径设置
os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.path.abspath(os.path.dirname(os.getcwd()))
os.path.abspath(os.path.join(os.getcwd(), ".."))

dataset ='qprop_small'

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

stop_words = set(stopwords.words('english'))

#获取句法依存关系对
rela_pair_count_str = {}
for doc_id in tqdm(range(0,len(yic_content_list))):
# for doc_id in range(len(yic_content_list)):
#     print(doc_id)
    words = yic_content_list[doc_id]
    words = words.split("\n")
    rela=[]
    for window in words:
        if window==' ':
            continue
        #构造rela_pair_count
        window = window.replace(string.punctuation, ' ')#把所有的标点符号替换为' '
        ws=window.split(" ")
        res = nlp.dependency_parse(window)#生成句法依存对/#"D:\ProgramData\Anaconda3\envs\python37\lib\site-packages\stanfordcorenlp\corenlp.py"
        for tuple in res:
            # rela.append(str(tuple[0]) + ', ' + str(tuple[1]))
            rela.append(str(tuple[0]) + ', ' + str(tuple[1])+', '+str(tuple[2]))
        for pair in rela:
            pair=pair.split(", ")
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
            if word_pair_str in rela_pair_count_str:
                rela_pair_count_str[word_pair_str] += 1
            else:
                rela_pair_count_str[word_pair_str] = 1
            # two orders
            # word_pair_str = pair[1]+ ',' + pair[0]
            # print(int(pair[2])-1,int(pair[1])-1)
            word_pair_str = ws[int(pair[2])-1] + ',' + ws[int(pair[1])-1]
            if word_pair_str in rela_pair_count_str:
                rela_pair_count_str[word_pair_str] += 1
            else:
                rela_pair_count_str[word_pair_str] = 1

#将rela_pair_count_str存成pkl格式
output1=open(output + '/{}_stan.pkl'.format(dataset),'wb')
pickle.dump(rela_pair_count_str, output1)