#!/usr/bin/env python
#coding=utf-8
###############################################
# File Name: add_label.py
# Author: Junliang Guo@USTC
# Email: guojunll@mail.ustc.edu.cn
###############################################

#python add_label.py deep_dictc.txt deep_matrix.txt ./data/citeseer/train_data.npy sample_size out_deep_matrix.txt
import sys
import numpy as np
from io import open
from collections import defaultdict as dd
import random
import os

dict_path = sys.argv[1]
co_path = sys.argv[2]
train_data_path = sys.argv[3]
label_sample_size = int(sys.argv[4])
matrix_out = sys.argv[5]

REMOVE = False
if co_path == matrix_out:
    REMOVE = True

emb_dic = {}
with open(dict_path, 'r') as f:
    k = 0
    for line in f:
        if k == 0:
            k += 1
            continue
        else:
            word = line.strip().split()[0]
            word = int(word)
            emb_dic[word] = k
            k += 1

train_data = np.load(train_data_path)
train_index = train_data[:, 0]
train_label = train_data[:, 1]
occ_m = np.zeros([len(emb_dic), len(emb_dic)])

with open(co_path, 'r') as f:
    for line in f:
        words = line.strip().split()
        words = [int(i) for i in words]
        if len(words) != 3:
            continue
        occ_m[words[0] - 1][words[1] - 1] += words[2]

if REMOVE:      #output is same as input. Remove input
    if os.path.exists(co_path):
        print 'remove previous co-occurrence matrix'
        os.remove(co_path)

pairs, label2inst = [], dd(list)
for i in xrange(len(train_index)):
    label2inst[train_label[i]].append(i)

for _ in range(label_sample_size):
    x1 = random.randint(0, len(train_index) - 1)
    label = train_label[x1]
    x2 = random.choice(label2inst[label])
    pairs.append([train_index[x1], train_index[x2]])

for pair in pairs:
    occ_m[emb_dic[pair[0]] - 1][emb_dic[pair[1]] - 1] += 1

with open(matrix_out, 'w', encoding = 'utf-8') as f:
    for i in xrange(occ_m.shape[0]):
        for j in xrange(occ_m.shape[1]):
            occ = occ_m[i][j]
            if occ != 0:
                occ = int(occ)
                out = [str(i + 1), str(j + 1), str(occ)]
                out = ' '.join(out)
                f.write(unicode(out) + '\n')
