#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
  ===> http request
  request the predict serving and get the probability
  ## notice: using [batch, 14*7] ID data as input ###
  ## important:
       1) where is suitable to trans Feature into IDs
       2) How to deal with the Feat never seen before
'''
import sys
import grpc
import math
import json
import time
import base64
import requests
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
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
  test_data_set = DataReadRaw( \
                  file_input = '../data/cur.day.testdata', \
                  seq_len    = seq_len, \
                  feat_len   = user_len + seq_len*item_len)
  test_data_set = test_data_set.data_read.values
  print (getNow(), 'test_set.shape', test_data_set.shape, type(test_data_set))
  pred_res= []
  real_res= []
  pred_label  = []
  pred_logits = []
  pred_logits_= []
  atime = []
  atime1= []

  for index in range(int(num_tests/concurrency)):
    cur_data    = test_data_set[index*concurrency:(index+1)*concurrency]
    print ('cur_data.shape=', cur_data.shape)
    cur_label   = cur_data[:, 0:seq_len]
    cur_feature = cur_data[:, seq_len:].reshape((-1, user_len+seq_len*item_len))
    datajson = {'signature_name':'predictByRaw', 'instances': []} # instances 表示行 # signature_name表示使用哪个 signature_def ##
    for j in range(len(cur_label)):
      datajson['instances'].append({'input':cur_feature[j,:].tolist()}) ## key->input 是在save_model时定义的输入名称 ##
    #print ('index=', index, 'cur_feature.shape=', cur_feature.shape, 'cur_feature=', cur_feature)
    btime = time.time()
    #r = requests.post('http://10.175.214.88:8501/v1/models/seq_model:predict', json=datajson)  ## here /v1/ 是要写的 ##
    r = requests.post('http://'+hostport+'/v1/models/seq_model:predict', json=datajson)  ## here /v1/ 是要写的 ##
    etime = time.time()
    atime.append(etime-btime)
    #r = requests.post('http://localhost:8501/models/seq_model:predict', json=datajson)
    try:
      response = json.loads(r.content.decode('utf-8'))
      predictions = response['predictions']
    except:
      print ('response error')
      continue
    atime1.append(etime-btime)
    for j in range(len(cur_label)):
      pred_res.append(predictions[j]['predict_prob'])
      pred_label.append(predictions[j]['predict_label'])
      pred_logits.append(predictions[j]['predict_logits'])
      #pred_logits_.append(predictions[j]['predict_logits_'])
      pred_logits_.append(predictions[j]['predict_logits_no'])
      print ('predict_prob', len(predictions[j]['predict_prob']))
      print ('predict_label', len(predictions[j]['predict_label']))
    real_res.append(list(cur_label))

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
  print ('avg-time[all]:', np.mean(atime))
  print ('avg-time[success]:', np.mean(atime1))
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
