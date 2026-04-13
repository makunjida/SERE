import tensorflow as tf
from layers import *
from metrics import *
import math
flags = tf.app.flags
FLAGS = flags.FLAGS

def calculate_Lkl(Asyn, Asem):
    # 使用 softmax 将 Asyn 和 Asem 转换为概率分布
    P_syn = tf.nn.softmax(Asyn)
    P_sem = tf.nn.softmax(Asem)

    # 计算 KL 散度
    kl_syn_sem = tf.reduce_sum(P_syn * tf.math.log(P_syn / P_sem), axis=1)
    kl_sem_syn = tf.reduce_sum(P_sem * tf.math.log(P_sem / P_syn), axis=1)

    # 计算 Lkl
    Lkl = tf.reduce_sum(tf.math.log(1 + 1 / (kl_syn_sem + kl_sem_syn)))
    return Lkl

class Model(object):
    def __init__(self, placeholders, features,transfer_lists, adj_lists, node_wgt_lists,**kwargs):
        self.adj_list = adj_lists
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = placeholders
        # self.input_dim=input_dim
        # self.transfer_list=transfer_list
        # self.adj_list=adj_list
        # self.node_wgt_list=node_wgt_list

        self.layers = []
        self.GCNlayers = []
        self.activations = []
        self.models=[GCN(placeholders, input_dim=features[2][1], logging=True, transfer_list = transfer_lists[0], adj_list = adj_lists[0], node_wgt_list = node_wgt_lists[0]),
                    GCN(placeholders, input_dim=features[2][1], logging=True, transfer_list = transfer_lists[1], adj_list = adj_lists[1], node_wgt_list = node_wgt_lists[1]),
                    GCN(placeholders, input_dim=features[2][1], logging=True, transfer_list = transfer_lists[2], adj_list = adj_lists[2], node_wgt_list = node_wgt_lists[2]),]
        self.attenlayer=Attention_Layer(placeholders,FLAGS.hidden,FLAGS.hidden)

        self.inputs = None
        self.output = None
        self.pred=None
        self.labels=None

        self.loss = 0
        self.accuracy = 0
        self.pre= 0
        self.f= 0
        self.rec= 0
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = None
        self.embed = None
        self.build()

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        outputs=[]
        for model in self.models:
            activations=[]
            GCNlayers=[]
            with tf.variable_scope(self.name):
                model._build()

            # Build sequential layer model
            activations.append(model.inputs)

            for i in range(len(model.layers)):
                layer = model.layers[i]
                hidden, pre_GCN = layer(activations[-1])
                GCNlayers.append(pre_GCN)
                if i >= FLAGS.coarsen_level and i < FLAGS.coarsen_level * 2:
                    hidden = hidden + GCNlayers[FLAGS.coarsen_level * 2-i - 1]
                activations.append(hidden)
            outputs.append(activations[-1])
        
        avg_out=self.attenlayer.forward(outputs[0],outputs[1],outputs[2])
        self.output = tf.nn.softmax(tf.layers.dense(avg_out, FLAGS.num_labels))


        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        # Weight decay loss
        # print(len(self.layers))
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.output, self.placeholders['labels'], #掩码的 Softmax 交叉熵损失
                                                  self.placeholders['labels_mask'])
        print(self.loss)

        #计算 Lkl 损失并加入到总损失中
        #Asyn = tf.convert_to_tensor(self.adj_list[1], dtype=tf.float32)  # 假设这是语法邻接矩阵
        #Asem = tf.convert_to_tensor(self.adj_list[2], dtype=tf.float32)  # 假设这是语义邻接矩阵
        
        #Lkl = calculate_Lkl(Asyn, Asem)
        #self.loss += FLAGS.alpha * Lkl  # 确保 FLAGS 中有 alpha 这一项

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.output, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.output, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN(object):
    def __init__(self, placeholders, logging,input_dim, transfer_list, adj_list, node_wgt_list, **kwargs):
        # super(GCN, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.logging=logging
        self.transfer_list = transfer_list
        self.adj_list = adj_list
        self.node_wgt_list = node_wgt_list

        self.W_node_wgt = tf.Variable(tf.random_uniform([FLAGS.max_node_wgt+1,FLAGS.node_wgt_embed_dim],
                     minval=-math.sqrt(6/(3*FLAGS.node_wgt_embed_dim+3*self.input_dim)),
                     maxval=math.sqrt(6/(3*FLAGS.node_wgt_embed_dim+3*self.input_dim))),
                     name="W_node_wgt")
        self.layers=[]
        #到这执行了_call()


    def _build(self):
        FCN_hidden_list = [FLAGS.hidden] * 100
        node_emb = tf.nn.embedding_lookup(self.W_node_wgt, self.node_wgt_list[0])
        #class GraphConvolution(Layer):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FCN_hidden_list[0],
                                            placeholders=self.placeholders,
                                            support = self.adj_list[0]* FLAGS.channel_num ,
                                            transfer = self.transfer_list[0],
                                            node_emb = node_emb,
                                            mod = 'input',
                                            layer_index = 0,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))  #G0

        for i in range(FLAGS.coarsen_level - 1):
            node_emb = tf.nn.embedding_lookup(self.W_node_wgt, self.node_wgt_list[i + 1])
            self.layers.append(GraphConvolution(input_dim=FCN_hidden_list[i],
                                            output_dim=FCN_hidden_list[i + 1],
                                            placeholders=self.placeholders,
                                            support = self.adj_list[i + 1] * FLAGS.channel_num,
                                            transfer = self.transfer_list[i + 1],
                                            node_emb = node_emb,
                                            mod = 'coarsen',
                                            layer_index = i + 1,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging) ) #Gi

        for i in range(FLAGS.coarsen_level, FLAGS.coarsen_level * 2):
            node_emb = tf.nn.embedding_lookup(self.W_node_wgt, self.node_wgt_list[2*FLAGS.coarsen_level - i])
            self.layers.append(GraphConvolution(input_dim=FCN_hidden_list[i - 1],
                                            output_dim=FCN_hidden_list[i],
                                            placeholders=self.placeholders,
                                            support = self.adj_list[2*FLAGS.coarsen_level - i] *FLAGS.channel_num,
                                            transfer = self.transfer_list[2*FLAGS.coarsen_level -1 -i],
                                            node_emb = node_emb,
                                            mod = 'refine',
                                            layer_index = i,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging) )#G?-1

        self.layers.append(GraphConvolution(input_dim=FCN_hidden_list[FLAGS.coarsen_level * 2 - 1],
                                            output_dim=FCN_hidden_list[FLAGS.coarsen_level * 2 - 1],
                                            placeholders=self.placeholders,
                                            support = self.adj_list[0] * FLAGS.channel_num,
                                            transfer = self.transfer_list[0],
                                            node_emb = 0,
                                            mod = 'output',
                                            layer_index = FLAGS.coarsen_level * 2,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

        #到这尚未执行_call()
    def predict(self):
        return tf.nn.softmax(self.outputs)




