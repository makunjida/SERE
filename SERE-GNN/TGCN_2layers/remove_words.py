from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, loadWord2Vec
from nltk.stem.porter import PorterStemmer
poter_stemme=PorterStemmer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)

dataset = 'qprop_small'

doc_content_list = []
f = open('../../data_tgcn/'+dataset+"/"+ dataset + '_corpus.txt', 'rb')#corpus
# f = open('data/wiki_long_abstracts_en_text.txt', 'r')
for line in f.readlines():
    doc_content_list.append(line.strip().decode('latin1'))
f.close()

word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
words_set=set()
for doc_content in doc_content_list:
    words=doc_content.split()
    for word in words:
        words_set.add(word)
print("there are {} words in this dataset.".format(len(words_set)))

for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        if word not in stop_words and word_freq[word] >= 5:
            word=poter_stemme.stem(word)
            doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)

f = open(f'../../data_tgcn/{dataset}/{dataset}.clean.txt', 'w')
#f = open('data/wiki_long_abstracts_en_text.qprop_small_ori.txt', 'w')
f.write(clean_corpus_str)
f.write("\n")
f.close()

#dataset = '20ng'
min_len = 10000
aver_len = 0
max_len = 0 

f = open(f'../../data_tgcn/{dataset}/{dataset}.clean.txt', 'r')
#f = open('data/wiki_long_abstracts_en_text.txt', 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    temp = line.split()
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
    if len(temp) > max_len:
        max_len = len(temp)
f.close()
aver_len = 1.0 * aver_len / len(lines)
print('min_len : ' + str(min_len))
print('max_len : ' + str(max_len))
print('average_len : ' + str(aver_len))

f_clean=open("../../data_tgcn/"+dataset+"/"+dataset+".clean.txt","r",encoding="utf-8").readlines()
f_clean_new=open("../../data_tgcn/"+dataset+"/"+dataset+"_new.clean.txt","w",encoding="utf-8")
f_label=open("../../data_tgcn/"+dataset+"/"+dataset+"_labels.txt","r",encoding="utf-8").readlines()
f_label_new=open("../../data_tgcn/"+dataset+"/"+dataset+"_new_labels.txt","w",encoding="utf-8")

ordinary_train_dict={}
ordinary_test_dict={}
i=-1
for line_corpus in f_clean:
    i=i+1
    if len(line_corpus.split(" ")) > 2:
        mode = f_label[i].split("\t")[1]
        label = f_label[i].split("\t")[-1].replace("-1","0")
        if mode=="train":
            ordinary_train_dict[line_corpus]=mode+"\t"+str(label)
        else:
            ordinary_test_dict[line_corpus] = mode + "\t" + str(label)

import random
dict_key_ls=list(ordinary_train_dict.keys())
random.shuffle(dict_key_ls)
new_train_dict={}
i=0
for key in dict_key_ls:
    label=ordinary_train_dict.get((key))
    new_train_dict[key]=str(i)+"\t"+label
    i=i+1

dict_key_ls=list(ordinary_test_dict.keys())
random.shuffle(dict_key_ls)
new_test_dict={}
for key in dict_key_ls:
    label=ordinary_test_dict.get((key))
    new_test_dict[key]=str(i)+"\t"+label
    i=i+1
print(len(new_train_dict)+len(new_test_dict))
for k,v in new_train_dict.items():
    f_clean_new.write(k)
    f_label_new.write(v)
for k,v in new_test_dict.items():
    f_clean_new.write(k)
    f_label_new.write(v)
# corpus=[]
# labels=[]
# i=-1
# for line_corpus in f_clean:
#     i = i + 1
#     if len(line_corpus.split(" "))>2:
#         if line_corpus not in corpus:
#             corpus.append(line_corpus)
#             # f_clean_new.write(line_corpus)
#             mode=f_label[i].split("\t")[1]
#             label=f_label[i].split("\t")[-1]
#             labels.append(mode+"\t"+label)
#             # f_label_new.write(mode+"\t"+label)