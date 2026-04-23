f=open("new2_labels.txt", "r", encoding="utf-8").readlines()
f_corpus=open("new2_corpus.txt","r",encoding="utf-8").readlines()
count_train,count_test=0,0
count_0=0
count_1=0
ls=set()
for i in range(len(f)):
    corpus=f_corpus[i][:-1]
    words=corpus.split()
    for word in words:
        ls.add(word)
    line=f[i]
    mode=line.split("\t")[1]
    if "train" in mode:
        count_train+=1
    else:
        count_test+=1
    label = line.split("\t")[-1]
    if "1" in label:
        count_1 += 1
    else:
        count_0 += 1


print(len(f),count_train,count_test,len(ls),count_1,count_0)

