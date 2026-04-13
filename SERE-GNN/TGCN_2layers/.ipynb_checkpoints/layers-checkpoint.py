from inits import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import scipy.sparse as sp
import numpy as np
import copy
import math
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph_only_2.txt of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs[0])
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, support, transfer, node_emb, mod, layer_index, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = support
        self.transfer = transfer
        self.node_emb = node_emb
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.mod = mod
        self.layer_index = layer_index
        self.output_dim = output_dim
        self.placeholders = placeholders

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            if self.mod == 'coarsen' or self.mod == 'refine':
                for i in range(len(self.support)):
                    self.vars['weights_' + str(i)] = glorot([input_dim + FLAGS.node_wgt_embed_dim, output_dim],
                                                            name='weights_' + str(i))
            else:
                for i in range(len(self.support)):
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        if self.mod == 'coarsen' or self.mod == 'refine':
            x = tf.concat([inputs, self.node_emb], 1)
            print('\nlayer_index ', self.layer_index + 1)
            print('input shape:   ', inputs.get_shape().as_list())
        elif self.mod == 'input' or self.mod == 'output':
            x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)

            else:
                pre_sup = self.vars['weights_' + str(i)]

            row = self.support[i][0][:, 0]
            col = self.support[i][0][:, 1]
            data = self.support[i][1]
            sp_suport = sp.csr_matrix((data, (row, col)), shape=self.support[i][2], dtype=np.float32)
            sp_suport_tensor = convert_sparse_matrix_to_sparse_tensor(sp_suport)
            support_ans = dot(sp_suport_tensor, pre_sup, sparse=True)
            supports.append(support_ans)

        supports = tf.convert_to_tensor(supports)
        supports = tf.transpose(supports, [1, 2, 0])
        output = tf.squeeze(tf.layers.conv1d(supports, 1, 1, use_bias=False))

        # bias
        if self.bias:
            output += self.vars['bias']
        output = self.act(output)

        gcn_output = output

        if self.mod == 'output':
            print('layer_index ', self.layer_index + 1)
            print('input shape:   ', inputs.get_shape().as_list())
            print('output shape:    ', output.get_shape().as_list())
            return output, gcn_output

        if self.mod == 'coarsen' or self.mod == 'input':
            transfer_opo = normalize(self.transfer.T, norm='l2', axis=1).astype(np.float32)
            transfer_opo = convert_sparse_matrix_to_sparse_tensor(transfer_opo)
            output = dot(transfer_opo, gcn_output, sparse=True)

        elif self.mod == 'refine':
            transfer_opo = convert_sparse_matrix_to_sparse_tensor(self.transfer.astype(np.float32))
            output = dot(transfer_opo, gcn_output, sparse=True)

        print('output shape:    ', output.get_shape().as_list())
        return output, gcn_output


class Attention_Layer():
    def softmax(self, X):
        e_X = tf.exp(X - tf.reduce_max(X, axis=1, keepdims=True))
        return e_X / tf.reduce_sum(e_X, axis=1, keepdims=True)

    def __init__(self, placeholders, output_dim, hidden_dim,dropout=0.5 ,**kwargs):
        super(Attention_Layer, self).__init__(**kwargs)

        self.placeholders=placeholders
        self.out_dim=output_dim
        self.hidden_dim=hidden_dim
        self.dropout=dropout
        self.loss=0
        self.acc=0

    def ffnn(self,X, num_hidden_layers, hidden_size, output_size, dropout=None):
        for i in range(num_hidden_layers):
            X = tf.layers.dense(X, hidden_size)#全连接层
            X = tf.nn.relu(X)
            if dropout is not None:
                X = tf.nn.dropout(X, dropout)
        output = tf.layers.dense(X, output_size)
        # print(output.shape)

        return output

    def compute_f(self,input1,input2,input3):
        
        weight_2 = tf.nn.sigmoid(
            self.ffnn(tf.concat([input1, input2], -1), FLAGS.FFNN_num_hidden_layers, self.hidden_dim, 1, self.dropout))
        weight_3 = tf.nn.sigmoid(
            self.ffnn(tf.concat([input1, input3], -1), FLAGS.FFNN_num_hidden_layers, self.hidden_dim, 1, self.dropout))
        weight = tf.expand_dims(tf.nn.softmax(tf.concat([weight_3, weight_2], -1), -1), 1)

        enhence_embeddings = tf.concat([tf.expand_dims(input3, 1), tf.expand_dims(input2, 1)], 1)
        enhence_embeddings = tf.squeeze(tf.matmul(weight, enhence_embeddings), 1)

        f = tf.sigmoid(self.ffnn(tf.concat([input1, enhence_embeddings], -1), FLAGS.FFNN_num_hidden_layers, self.hidden_dim, self.out_dim))

        return f,enhence_embeddings



    def forward(self, input1, input2, input3):
        f1, enhence_embeddings1 = self.compute_f(input1, input2, input3)
        f2, enhence_embeddings2 = self.compute_f(input2, input1, input3)
        f3, enhence_embeddings3 = self.compute_f(input3, input1, input2)

        layer_out1 = f1 * input1 + (1 - f1) * enhence_embeddings1
        layer_out2 = f2 * input2 + (1 - f2) * enhence_embeddings2
        layer_out3 = f3 * input3 + (1 - f3) * enhence_embeddings3
        
        
        
        input_tensor1 = tf.expand_dims(tf.stack((layer_out1, layer_out2,layer_out3), axis=0), axis=0)
        avg_tensor = [1, 3, 1, 1]
        avg_strides = [1, 3, 1, 1]
        avg_out1 = tf.squeeze(tf.nn.avg_pool(value=input_tensor1, ksize=avg_tensor, strides=avg_strides, padding="SAME"))
        

        # 将增强后的嵌入合并为一个张量
        input_tensor2 = tf.stack((layer_out1, layer_out2, layer_out3), axis=0)
    
        print(layer_out1.shape)

        # 自注意力机制
        query = input_tensor2  # (3, batch_size, feature_dim)
        key = input_tensor2    # (3, batch_size, feature_dim)
        value = input_tensor2  # (3, batch_size, feature_dim)

        # 计算注意力分数
        attention_scores = tf.matmul(query, key, transpose_b=True)  # (3, 3)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (3, 3)
        
          # 计算加权值
        attention_output = tf.matmul(attention_weights, value)  # (3, batch_size, feature_dim)
            # 保持输出的形状与平均池化一致，reshape 为 (batch_size, feature_dim)
            # 注意：tf.reduce_sum 通过对第一个维度求和来合并三个特征向量
        avg_out2 = tf.reduce_sum(attention_output, axis=0)  # (batch_size, feature_dim)
        
        avg_out = avg_out1  # 在最后一个维度上拼接

        print('最终输出')
        print(avg_out.shape)
    
        return avg_out
