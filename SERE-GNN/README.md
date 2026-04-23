
# Reproduing Results

#### 1. Build three graphs

Run TGCN_2layers/build_graph_tgcn.py

#### 2. Training

Run TGCN_2layers/train.py

# Example input data

1. `/data_tgcn/mr/build_train/mr.clean.txt` indicates document names, training/test split, document labels. Each line is for a document.

2. `/data_tgcn/mr/build_train/mr.txt` contains raw text of each document.

3. `/data_tgcn/mr/stanford/mr_pair_stan.pkl` contains all syntactic relationship word pairs for the dataset.

4. `/data_tgcn/mr/build_train/mr_semantic_0.05.pkl` contains all semantic relationship word pairs for the dataset.

  
构建三个图：
运行 TGCN_2layers/build_graph_tgcn.py

训练：
运行 TGCN_2layers/train.py

/data_tgcn/mr/build_train/mr.clean.txt：指示文档名称、训练/测试拆分和文档标签的文件。每一行对应一个文档。

/data_tgcn/mr/build_train/mr.txt：包含每个文档的原始文本。

/data_tgcn/mr/stanford/mr_pair_stan.pkl：包含数据集的所有句法关系词对。

/data_tgcn/mr/build_train/mr_semantic_0.05.pkl：包含数据集的所有语义关系词对。
