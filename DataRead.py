#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def getNow():
  return '['+str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(time.time()))))+']'

class DataRead:
  def __init__(self, file_input='./tmp', seq_len=14, feat_len=98):
    self.file_input = file_input
    self.seq_len    = seq_len
    self.feat_len   = feat_len
    self.data_read  = self.pdreaddata()
  def pdreaddata(self):   
    data_read  = pd.read_table(self.file_input, sep='\t', dtype=np.int32, \
         names = ['label_'+str(i) for i in range(self.seq_len)] + \ 
                 ['feat_'+str(i) for i in range(self.feat_len)])
    #     names = ['label_'+str(i) for i in range(14)] + \
    #             ['feat_'+str(i) for i in range(98)])

    print getNow(), 'read data :', data_read.shape
    return data_read
  def shuffle(self):
    shuf_data = shuffle(self.data_read)
    print getNow(), 'shuf data :', shuf_data.shape
    return shuf_data
