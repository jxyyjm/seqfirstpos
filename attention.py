#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

def MultiHeadsAttention(query_layer, key_layer, value_layer):
    attend_z  = tf.matmul(query_layer, tf.transpose(key_layer, [0, 1, 3, 2]))
    size_perH = tf.cast(query_layer.get_shape().as_list()[-1], tf.float32)
    scaled_z  = tf.multiply(attend_z, 1/tf.sqrt(size_perH))
    softmax_z = tf.nn.softmax(scaled_z, axis=3) # equal dim=-1 # or default #
    multi_att = tf.matmul(softmax_z, value_layer)
    return multi_att

def MultiHeadsDenseLayer(tensor_input, w, attention_size, heads=1):
    ## input.shape = [batch, seq, emb_size] ##
    _, seq_length, emb_size = tensor_input.get_shape().as_list()
    size_per_head      = int(attention_size/heads)
    tensor_input_2d    = tf.reshape(tensor_input, [-1, emb_size])
    tensor_input_layer = tf.matmul(tensor_input_2d, w)
    tensor_input_layer = tf.reshape(tensor_input_layer, [-1, seq_length, heads, size_per_head])
    tensor_input_layer = tf.transpose(tensor_input_layer, [0, 2, 1, 3]) 
    return tensor_input_layer

def fcn_self_attention(query, key, value, attention_size, heads=1):
    hidden_dim = query.get_shape().as_list()[-1]
    with tf.variable_scope('query_layer', reuse=tf.AUTO_REUSE):
        w_query = tf.get_variable('w_query', dtype=tf.float32, shape=[hidden_dim, attention_size],
                                  regularizer = tf.contrib.layers.l2_regularizer(0.01),
                                  initializer = tf.contrib.layers.xavier_initializer())
        query_layer = MultiHeadsDenseLayer(query, w_query, attention_size, heads)
    with tf.variable_scope('key_layer', reuse=tf.AUTO_REUSE):
        w_key = tf.get_variable('w_key', dtype=tf.float32, shape=[hidden_dim, attention_size],
                                  regularizer = tf.contrib.layers.l2_regularizer(0.01),
                                  initializer = tf.contrib.layers.xavier_initializer())
        key_layer = MultiHeadsDenseLayer(key, w_key, attention_size, heads)
    with tf.variable_scope('value_layer', reuse=tf.AUTO_REUSE):
        w_value = tf.get_variable('w_value', dtype=tf.float32, shape=[hidden_dim, attention_size],
                                  regularizer = tf.contrib.layers.l2_regularizer(0.01),
                                  initializer = tf.contrib.layers.xavier_initializer())
        value_layer = MultiHeadsDenseLayer(value, w_value, attention_size, heads)
    multiH_attention = MultiHeadsAttention(query_layer, key_layer, value_layer)
    return multiH_attention
