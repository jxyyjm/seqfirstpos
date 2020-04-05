#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
  ===> grpc request 
  request the predict serving and get the probability
  ## notice: using [batch, 14*7] ID data as input ###
  ## important:
       1) where is suitable to trans Feature into IDs
       2) How to deal with the Feat never seen before
'''
import sys 
import grpc
import math
import time
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from DataReadRaw import DataReadRaw, getNow

tf.app.flags.DEFINE_integer('concurrency', 1, 'max num of concur inf req')
tf.app.flags.DEFINE_integer('num_tests', 10000, 'num of test-case')
tf.app.flags.DEFINE_string('server', '', 'PredServing host:port')
tf.app.flags.DEFINE_string('work_dir', './tmp/', 'work directory')
FLAGS = tf.app.flags.FLAGS
seq_len = 14
item_len = 4 
user_len = 1 
def do_inference(hostport, work_dir, concurrency, num_tests):
  #test_data_set = DataRead('../data/seq.train.data.20191127.part')
  #test_data_set = DataRead('../data/seq.test.data.20191128.part')
  test_data_set = DataReadRaw( \
                  file_input = '../data/cur.day.testdata', \
                  seq_len    = seq_len, \
                  feat_len   = user_len + seq_len*item_len)
  test_data_set = test_data_set.data_read.values
  print (getNow(), 'test_set.shape', test_data_set.shape, type(test_data_set))
  channel = grpc.insecure_channel(hostport)
  stub    = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  pred_res= []
  real_res= []
  pred_label  = []
  pred_logits = []
  pred_logits_= []
  atime = []
  atime1 = []

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'seq_model' ## model name
  request.model_spec.signature_name = 'predictByRaw' ## signature_def_map ##

  for index in range(int(num_tests/concurrency)):
    cur_data    = test_data_set[index*concurrency:(index+1)*concurrency]
    print ('cur_data.shape=', cur_data.shape)
    cur_label   = cur_data[:, 0:seq_len]
    cur_feature = cur_data[:, seq_len:].reshape((-1, user_len+seq_len*item_len))
    #print ('index=', index, 'cur_feature.shape=', cur_feature.shape, 'cur_feature=', cur_feature)
    request.inputs['input'].CopyFrom(
      tf.contrib.util.make_tensor_proto(values=cur_feature, dtype=tf.string, shape=cur_feature.shape))
    btime = time.time()
    response = stub.Predict(request, 5.0)
    etime = time.time()
    atime.append(etime-btime)

    try:
      results  = tf.contrib.util.make_ndarray(response.outputs['predict_prob'])
      results1 = tf.contrib.util.make_ndarray(response.outputs['predict_label'])
      results2 = tf.contrib.util.make_ndarray(response.outputs['predict_logits'])
      #results3 = tf.contrib.util.make_ndarray(response.outputs['predict_logits_'])
      results3 = tf.contrib.util.make_ndarray(response.outputs['predict_logits_no'])
    except: continue
    atime1.append(etime-btime)

    #print ('index=', index, 'result=', results)
    pred_res.append(list(results))
    real_res.append(list(cur_label))
    pred_label.append(list(results1))
    pred_logits.append(list(results2))
    pred_logits_.append(list(results3))

  pred_res = np.array(pred_res).reshape((-1, 14))
  real_res = np.array(real_res).reshape((-1, 14))
  pred_label = np.array(pred_label).reshape((-1,14))
  pred_logits = np.array(pred_logits).reshape((-1,14))
  pred_logits_ = np.array(pred_logits_).reshape((-1,14))

  count, e_count, p_count, n_count = 0, 0, 0, 0
  e_p_count, e_n_count = 0, 0
  index_zero_p_count= 0
  index_zero_p_pred = 0
  index_zero_n_count= 0
  index_zero_n_pred = 0

  for r in range(pred_res.shape[0]):
    for z in range(14):
      count += 1
      if real_res[r][z] == pred_label[r][z]:
        e_count += 1
      if real_res[r][z] == 1:
        p_count += 1
        if real_res[r][z] == pred_label[r][z]:
          e_p_count += 1
      if real_res[r][z] == 0:
        n_count += 1
        if real_res[r][z] == pred_label[r][z]:
          e_n_count += 1

    if real_res[r][0] == 1:
      index_zero_p_count += 1
      if pred_label[r][0] == 1:
        index_zero_p_pred += 1
    else:
      index_zero_n_count += 1
      if pred_label[r][0] == 0:
        index_zero_n_pred += 1

    print ('real_label', '\t'.join([str(int(x)) for x in real_res[r]]))
    print ('pred_label', '\t'.join([str(int(x)) for x in pred_label[r]]))
    print ('pred_logits', '\t'.join([str(round(x, 3)) for x in pred_logits[r]]))
    print ('pred_prob' , '\t'.join([str(round(x, 3)) for x in pred_res[r]]))
    print ('pred_logits_', '\t'.join([str(round(x, 3)) for x in pred_logits_[r]]))
    print ('pred_logits_[prob]', '\t'.join([str(round(1/(1+math.exp(-x)),4)) for x in pred_logits_[r]]))
    print ('==='*40)
  print ('equal_count/count=', round(e_count/count*1.0, 4), e_count, count)
  print ('positive equal_count/count=', round(e_p_count/p_count*1.0, 4), e_p_count, p_count)
  print ('negative equal_count/count=', round(e_n_count/n_count*1.0, 4), e_n_count, n_count)
  print ('index-0, pred-1/label-1', round(index_zero_p_pred/index_zero_p_count*1.0, 4), index_zero_p_pred, index_zero_p_count)
  print ('index-0, pred-0/label-0', round(index_zero_n_pred/index_zero_n_count*1.0, 4), index_zero_n_pred, index_zero_n_count)
  print ('ava-time[all]:', np.mean(atime))
  print ('ava-time[success]:', np.mean(atime1))
def main(_):
  if FLAGS.num_tests > 10001:
    print ('num_test should not be gt 10k')
    return
  if not FLAGS.server:
    print ('please specify server host:port')
    return
  do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency, FLAGS.num_tests)

if __name__=='__main__':
  tf.app.run()
