from transformers import BertModel, BertTokenizer
import torch
from progressbar import *
from tqdm import tqdm
use_cuda = torch.cuda.is_available()
dataset="ptc"

device = torch.device('cuda' if use_cuda else 'cpu')
print(device)
def get_bert_embedding(sentences, similarity_threshold = 0.5):
  pbar = ProgressBar(widgets=['Getting wordwise similarity ', Percentage(), ' ', Bar(),
        ' ', ETA(), ' '], maxval=len(sentences)).start()
  total_set = {}
  valid_set={}
  tokenizer = BertTokenizer.from_pretrained("D:/ProgramData/bert-base-uncased")
  # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  model = BertModel.from_pretrained("D:/ProgramData/bert-base-uncased").to(device)
  # model = BertModel.from_pretrained("bert-base-uncased").to(device)
  count = 0
  for i in tqdm(range(0,len(sentences))):
    # print(i)
    s=sentences[i]
  # for s in sentences:
    pbar.update(count)
    encoded_input = tokenizer(s, return_tensors='pt').to(device)
    output = model(**encoded_input)
    aa = torch.squeeze(output.last_hidden_state,dim=0)
    each = aa[1:-1,:]
    split_s = s.split()
    for word in range(0,len(split_s)):
      for another in range(0,len(split_s)):
        w1 = split_s[word]
        w2 = split_s[another]
        score = torch.cosine_similarity(each[word].view(1,-1),each[another].view(1,-1),dim=1)#计算相似度(同一个句子不同单词之间的相似度)
        key = ''+w1+','+w2
        if key not in total_set.keys():
          total_set[key] = 1
        else:
          total_set[key]+=1
        if score.data>similarity_threshold:
          if key not in valid_set.keys():
            valid_set[key] = 1
          else:
            valid_set[key] += 1
    count +=1
  pbar.finish()
  return total_set,valid_set

# DO NOT TOKENIZE YOUR SENTENCE!!!!!! [[sentence0],[sentence1],[sentence2],[sentence3],.......]
sentences=[]
file=open(f'../../data_tgcn/{dataset}/{dataset}_new.clean.txt', "r", encoding="utf-8").readlines()
for line in file:
  sentences.append(line[:-1])
total_set,valid_set = get_bert_embedding(sentences)
# print(total_set)
# print(valid_set)
import pickle

save_path=f'../../data_tgcn/{dataset}/bert'
if not os.path.exists(save_path):
  os.mkdir(save_path)
f = open(os.path.join(save_path,'{dataset}_semantic.pkl'), 'wb')
pickle.dump(valid_set,f)
f.close()