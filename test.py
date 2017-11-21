#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################
# File Name: test.py
# Author: Junliang Guo@USTC
# Email: guojunll@mail.ustc.edu.cn
###############################################

import numpy as np
from scipy import sparse
from io import open
import json
import random
import sys
from utils import load_pdata
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from itertools import izip

#python embedding_file dict dataset classes

mat_variables = loadmat(sys.argv[1])
W_matrix = mat_variables['W_t']

dic_p = sys.argv[2]
emb_dic = {}
with open(dic_p, 'r') as f:
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
classes = int(sys.argv[4])
_, _, train_data, test_data = load_pdata(sys.argv[3])
index = test_data[:, 0]
test_l = test_data[:, 1]
test_label = []
for i in xrange(test_data.shape[0]):
    temp = [0] * classes
    temp[test_data[i][1] - 1] += 1
    test_label.append(temp)
test_label = np.array(test_label)   #1000 * 6

train_index = train_data[:, 0]
train_l = train_data[:, 1]
train_label = []
for i in xrange(train_data.shape[0]):
    temp = [0] * classes
    temp[train_data[i][1] - 1] += 1
    train_label.append(temp)
train_label = np.array(train_label)   #120 * 6
test_in = []
train_in = []

W = np.transpose(W_matrix)
for i in index:
    zeros = [0] * W.shape[1]
    if i in emb_dic:
        emb_id = emb_dic[i]
        if emb_id <= W.shape[0]:
            emb_v = W[emb_id - 1, :]
            test_in.append(emb_v)
        else:
            test_in.append(zeros)
    else:
        test_in.append(zeros)

for i in train_index:
    zeros = [0] * W.shape[1]
    if i in emb_dic:
        emb_id = emb_dic[i]
        if emb_id <= W.shape[0]:
            emb_v = W[emb_id - 1, :]
            train_in.append(emb_v)
        else:
            train_in.append(zeros)
    else:
        train_in.append(zeros)


test_in = np.asarray(test_in)
train_in = np.asarray(train_in)

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels
y_train_ = sparse.coo_matrix(train_label)
y_train = [[] for x in xrange(y_train_.shape[0])]
cy =  y_train_.tocoo()
for i, j in izip(cy.row, cy.col):
    y_train[i].append(j)

assert sum(len(l) for l in y_train) == y_train_.nnz

y_test_ = sparse.coo_matrix(test_label)

y_test = [[] for x in xrange(y_test_.shape[0])]
cy =  y_test_.tocoo()
for i, j in izip(cy.row, cy.col):
    y_test[i].append(j)
y_train = np.array(y_train)
clf = TopKRanker(LogisticRegression())
clf.fit(train_in, y_train)

top_k_list = [len(l) for l in y_test]
preds = clf.predict(test_in, top_k_list)
acc = accuracy_score(y_test, preds)
print 'acc: %.3f' % acc