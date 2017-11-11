#!/usr/bin/env python
#coding=utf-8
###############################################
# File Name: p_feature.py
# Author: Junliang Guo
# mail: guojunliangkk@gmail.com
# Created Time: Fri 10 Feb 2017 04:45:12 PM CST
# Description: 
###############################################

import sys
from scipy import sparse as sp
from scipy import io as sio
import numpy as np
from io import open
#python p_feature.py in_file out_file dict_file

feature_in = sys.argv[1]
out_p = sys.argv[2]
dic_p = sys.argv[3]

dic = {}
k = 0
with open(dic_p, 'r') as f:
    for line in f:
        #print line
        if k == 0:
            k += 1
        else:
            node = line.strip().split()[0]
            dic[k] = node
            k += 1
#print len(dic)
features = sio.loadmat(feature_in)['features']
#print features[int(dic[11])]
features = features.todense()
#print features.shape
temp_m = np.zeros([len(dic), features.shape[1]])
#print temp_m.shape
for i in xrange(temp_m.shape[0]):
    temp_m[i] = features[int(dic[i + 1])]
temp_m = sp.csr_matrix(temp_m, dtype = 'double')
#print temp_m[10]
sio.savemat(out_p, {'features': temp_m})




