from __future__ import division
from __future__ import print_function

import os
import random
import time
import copy
import tensorflow as tf
from models import Model
from sklearn import metrics
from layers import Attention_Layer
from utils import *
from coarsen import *
from metrics import *

dataset = 'ptc'
print(dataset)

os.path.abspath(os.path.dirname(os.path.dirname(__file__)))#获取当前文件的上两级目录的绝对路径
os.path.abspath(os.path.dirname(os.getcwd()))#获取当前工作目录的父目录的绝对路径。
os.path.abspath(os.path.join(os.getcwd(), ".."))
f_file = os.sep.join(['..','..', 'data_tgcn', dataset, 'build_train'])#构建一个文件路径 f_file

# Set random seed每次实验生成一个固定的随机数
seed = random.randint(1, 200)
# seed=148
np.random.seed(seed)
tf.set_random_seed(seed)
best_test_Acc=0
best_test_epoch=0

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed','DataSet_Misinfo'
flags.DEFINE_string('dataset', dataset, 'CAAprop string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')#学习率
flags.DEFINE_integer('FFNN_num_hidden_layers', 2,'the number of the hidden layers of ffnn function')#前馈神经网络的隐藏层数
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')#训练轮数
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')#隐藏层的单元数
flags.DEFINE_integer('num_labels', 2, 'Number of units in mlp2 output1.')
flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('hidden', 32, 'Number of units in hidden layer 1.')#维度设置为32
flags.DEFINE_integer('coarsen_level',2, 'Maximum coarsen level.')#粗话层数
flags.DEFINE_integer('max_node_wgt', 50, 'Maximum node_wgt to avoid super-node being too large.')
flags.DEFINE_integer('node_wgt_embed_dim', 8, 'Number of units for node weight embedding.')#节点权重嵌入的单位数
flags.DEFINE_integer('channel_num', 4, 'Number of channels')
flags.DEFINE_float('weight_decay', 5e-4,
                   'Weight for L2 loss on embedding matrix.')  #L2权衰减为1e−7
flags.DEFINE_integer('early_stopping', 0,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')#最大切比雪夫多项式度

##加载语料库的数据，并将加载的数据分配给了多个变量。具体来说，它加载了邻接矩阵（adj,adj1, adj2）、特征矩阵（features）以及训练、验证和测试集的标签（y_train、y_val、y_test）以及掩码（train_mask、val_mask、test_mask）。
adj, adj1, adj2, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size, test_size = load_corpus(
    f_file, FLAGS.dataset)


# features = sp.identity(features.shape[0])  # featureless
# Some preprocessing预处理
features = preprocess_features(features)

graphs=[]
mappings=[]
graph, mapping = read_graph_from_adj(adj, FLAGS.dataset,"graph")
graphs.append(graph)
mappings.append(mapping)
graph, mapping= read_graph_from_adj(adj1, FLAGS.dataset,"graph1")
graphs.append(graph)
mappings.append(mapping)
graph, mapping = read_graph_from_adj(adj2, FLAGS.dataset,"graph2")
graphs.append(graph)
mappings.append(mapping)

graph_name=['sequential-based','semantic-based','syntactic-based']
for i in range(0,3):
    print(graph_name[i]+' graph total nodes:', graphs[i].node_num)
print("\n")
# Step-1: Graph Coarsening.
transfer_lists=[]
adj_lists=[]
node_wgt_lists=[]

for j in range(0,3):
    graph = graphs[j]
    original_graph = graph
    transfer_list = []
    adj_list = [copy.copy(graph.A)]
    node_wgt_list = [copy.copy(graph.node_wgt)]
    for i in range(FLAGS.coarsen_level):
        match, coarse_graph_size = generate_hybrid_matching(FLAGS.max_node_wgt, graph)
        coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)
        transfer_list.append(copy.copy(graph.C))#(24065, 40)	1#变换矩阵
        graph = coarse_graph
        #在此之前是未粗化操作图的数据
        #在此之后是粗化后图的数据
        adj_list.append(copy.copy(graph.A))#(5330, 10)	2.0#邻接矩阵
        node_wgt_list.append(copy.copy(graph.node_wgt))#[2 2 2 ... 1 1 1]#节点权重向量
        print('There are %d nodes in the %d coarsened %s graph_only_2.txt' %(graph.node_num, i+1,graph_name[j]))
    transfer_lists.append(transfer_list)
    adj_lists.append(adj_list)
    node_wgt_lists.append(node_wgt_list)
    print("\n")
for adj_list in adj_lists:
    for i in range(len(adj_list)):
        adj_list[i] = [preprocess_adj(adj_list[i])]

if FLAGS.model == 'gcn':
    support1 = [preprocess_adj(adj)]
    support2=[preprocess_adj(adj1)]
    support3 = [preprocess_adj(adj2)]
    num_supports = 1
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
# Create model
model=Model(placeholders, features=features, transfer_lists = transfer_lists, adj_lists = adj_lists, node_wgt_lists = node_wgt_lists)

sess = tf.Session()
# Define model evaluation function
def evaluate(features, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, labels, mask, placeholders)
    loss,acc,pred,labels=sess.run([model.loss,model.accuracy,model.pred,model.labels], feed_dict=feed_dict_val)
    return loss,acc,pred,labels,(time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

cost_train = []
acc_train = []

cost_test = []
acc_test = []
f1_test=[]
precision_test=[]
recall_test=[]

best_fcn = 0

# f_result=open("../result/"+str(dataset)+"_"+str(FLAGS.dropout)+"_"+str(FLAGS.learning_rate)+"_"+str(FLAGS.coarsen_level)+"_"+str(FLAGS.FFNN_num_hidden_layers)+"_"+bert_mode+".txt","w",encoding="utf-8")
# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary

    feed_dict = construct_feed_dict(features, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op,model.loss,model.accuracy,model.output], feed_dict=feed_dict)#返回numpy.ndarray

    #Validation
    eval_loss,eval_acc,eval_pred,eval_labels, duration=evaluate(features, y_val, val_mask, placeholders)
    # print(eval_loss,eval_acc)
    cost_val.append(eval_loss)

    test_loss, test_acc, pred, labels, test_duration = evaluate(features, y_test, test_mask, placeholders)


    cost_train.append(outs[1])
    acc_train.append(outs[2])

    reslut = ("Epoch: " + str('%04d' % (epoch + 1)) + " train_loss= " + str(
        "{:.5f}".format(outs[1])) + " train_acc= " + str("{:.5f}".format(outs[2])) +
              " val_loss= " + str("{:.5f}".format(eval_loss)) + " val_acc= " + str(
                "{:.5f}".format(eval_acc)) + " test_loss= " + str("{:.5f}".format(test_loss)) + " test_acc= " + str(
                "{:.5f}".format(test_acc)) + " time= " + str("{:.5f}".format(time.time() - t)))
    print(reslut)
    # f_result.write(reslut+"\n")
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break
print("Optimization Finished!")

# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, y_test, test_mask, placeholders)
test_result="Test set results:"+ "cost="+str("{:.5f}".format(test_cost))+\
            "accuracy="+str("{:.5f}".format(test_acc))+ "time="+str("{:.5f}".format(test_duration))
# f_result.write(test_result)
print(test_result)

test_pred = []
test_labels = []
print(len(test_mask))
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(test_labels, test_pred, digits=4,zero_division=0))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro',zero_division=0))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro',zero_division=0))

