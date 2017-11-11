#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################
# File Name: utils.py
# Author: Junliang Guo@USTC
# Email: guojunll@mail.ustc.edu.cn
###############################################

import argparse
import random
from scipy import sparse as sp
from scipy import io as sio
import networkx as nx
import numpy as np
import pickle as pkl

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def process_feature(f_before,  dic_p):

    dic = {}
    k = 0
    with open(dic_p, 'r') as f:
        for line in f:
            if k == 0:
                k += 1
            else:
                node = line.strip().split()[0]
                dic[k] = node
                k += 1
    features = f_before.todense()
    temp_m = np.zeros([features.shape[0], features.shape[1]])
    for i in xrange(temp_m.shape[0]):
        temp_m[i] = features[int(dic[i + 1])]
    f_after = sp.csr_matrix(temp_m)
    return f_after

def load_pdata(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in xrange(len(names)):
        objects.append(pkl.load(open("./data/ind.{}.{}".format(dataset_str, names[i]))))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))

    train_mask = sample_mask(idx_train, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    train_out = []
    for i in idx_train:
        ll = y_train[i].tolist()
        ll = ll.index(1) + 1
        train_out.append([i, ll])
    train_out = np.array(train_out)
    np.random.shuffle(train_out)

    test_out = []
    for i in idx_test:
        ll = y_test[i].tolist()
        ll = ll.index(1) + 1
        test_out.append([i, ll])
    test_out = np.array(test_out)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_mask = int(np.floor(edges.shape[0] / 10.))

    return graph, features, train_out, test_out
