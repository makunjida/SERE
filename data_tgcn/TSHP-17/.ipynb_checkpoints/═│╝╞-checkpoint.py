f=open("proppy_sm_labels.txt", "r", encoding="utf-8").readlines()
count_0=0
count_1=0

for line in f:
    label=line.split("\t")[-1]
    if "1" in label:
        count_1+=1
    else:
        count_0+=1
print(count_0,count_1)
