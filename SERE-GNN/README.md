# TensorGCN

The implementation of [TensorGCN](https://arxiv.org/pdf/2001.05313.pdf) in paper:

Liu X, You X, Zhang X, et al. Tensor graph convolutional networks for text classification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(05): 8409-8416.


# Require

Python 3.6

Tensorflow >= 1.11.0


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


# Semantic-based graph
we propose a LSTM-based method to construct a semantic-based graph from text documents. There are three main steps:
- Step 1: Train a LSTM on the training data of the given task (e.g. text classification here).
- Step 2: Get semantic features/embeddings with LSTM for all words in each document/sentence of the corpus.
- Step 3: Calculate word-word edge weights based on word semantic embeddings over the corpus.The calculation formula can be found in formula (3) in the paper.


# Syntactic-based graph
- Step 1: We utilize stanford CoreNLP parser to extract dependency between words. You can learn how to use the toolkit through [this website.](https://www.pianshen.com/article/8433287443/)
- Step 2: Get syntactic relationship word pairs for the dataset by :
  Run TGCN1_2layers/get_syntactic_relationship.py. 
  
  
这个项目实现了论文中提出的TensorGCN模型，用于文本分类任务。要重新生成结果，按照以下步骤操作：

构建三个图：
运行 TGCN_2layers/build_graph_tgcn.py

训练：
运行 TGCN_2layers/train.py

示例输入数据包括：

/data_tgcn/mr/build_train/mr.clean.txt：指示文档名称、训练/测试拆分和文档标签的文件。每一行对应一个文档。

/data_tgcn/mr/build_train/mr.txt：包含每个文档的原始文本。

/data_tgcn/mr/stanford/mr_pair_stan.pkl：包含数据集的所有句法关系词对。

/data_tgcn/mr/build_train/mr_semantic_0.05.pkl：包含数据集的所有语义关系词对。

语义图构建方法：

第一步：在给定任务的训练数据上训练一个LSTM模型（例如，在这里进行文本分类）。
第二步：使用LSTM为语料库中每个文档/句子中的所有单词获取语义特征/嵌入。
第三步：基于语料库中单词语义嵌入计算单词之间的边权重。计算公式可以在论文中的公式（3）中找到。
句法图构建方法：

第一步：我们利用Stanford CoreNLP解析器提取单词之间的依赖关系。您可以通过此网站了解如何使用该工具包。
第二步：通过运行 TGCN1_2layers/get_syntactic_relationship.py 获取数据集的句法关系词对。