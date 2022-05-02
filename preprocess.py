#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:52:18 2022

@author: pnaddaf
"""

import numpy as np
import ast
import pickle



save_path = '/localhome/pnaddaf/Desktop/parmis/SEAl_miror/datasets_LLGF/'
dataset = 'fb237_v1'

ind = ''

x_path = '/localhome/pnaddaf/Desktop/parmis/grail-master/data/FB15K237/entity2vec.txt'
train_path = "/localhome/pnaddaf/Desktop/parmis/grail-master/data/" + dataset + "/train.txt"
test_path = "/localhome/pnaddaf/Desktop/parmis/grail-master/data/" + dataset + "/test.txt"
val_path = "/localhome/pnaddaf/Desktop/parmis/grail-master/data/" + dataset + "/valid.txt"

nodes_dict_path = "/localhome/pnaddaf/Desktop/parmis/grail-master/data/FB15K237/entities.dict"


def read_edges(edge_path):
    edge_list = []
    with open(edge_path) as f:
        for line in f:
            line = line.split("\t")
            source = nodes_dict[line[0]]
            target = nodes_dict[line[2][:-1]]
            edge_list.append([source, target])
            edge_list.append([target, source])
            edge_list.append([target, target])
            edge_list.append([source, source])
            all_nodes.add(source)
            all_nodes.add(target)
    return edge_list
     
            

def get_false_edges(test_edges, train_edges, val_edges ):
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    
    edges_all = np.asarray( test_edges + train_edges + val_edges,  dtype=np.float32)
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, len(all_nodes))
        idx_j = np.random.randint(0, len(all_nodes))
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue

        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, len(all_nodes))
        idx_j = np.random.randint(0, len(all_nodes))
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, len(all_nodes))
        idx_j = np.random.randint(0, len(all_nodes))
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_i, idx_j], np.array(val_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])


    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    
    return train_edges_false, test_edges_false, val_edges_false


nodes_dict = dict()
with open(nodes_dict_path) as f:
    for line in f:
       line = line.split("\t")
       nodes_dict[line[1][:-1]] = int(line[0])





x = []
with open(x_path) as f:
    for line in f:
        x.append([float(i) for i in line.split("\t")[:-1]])

np.save(save_path + 'LLGF_FB' + ind + '_x.npy', np.array(x))
       
        
        
all_nodes = set()


train_edges = read_edges(train_path)
print(len(all_nodes))
test_edges = read_edges(test_path)
print(len(all_nodes))
val_edges = read_edges(val_path)
print(len(all_nodes))
train_edges_false, test_edges_false, val_edges_false = get_false_edges(test_edges, train_edges, val_edges )

np.save(save_path + 'LLGF_FB' + ind + '_train_pos.npy', np.array(train_edges))

np.save(save_path + 'LLGF_FB' + ind + '_test_pos.npy', np.array(test_edges))
np.save(save_path + 'LLGF_FB' + ind + ' _test_neg.npy', np.array(test_edges_false))

np.save(save_path + 'LLGF_FB' + ind+ '_val_pos.npy', np.array(val_edges))
np.save(save_path + 'LLGF_FB' + ind + '_val_neg.npy', np.array(val_edges_false))


cora_train = np.load('/localhome/pnaddaf/Desktop/parmis/SEAl_miror/datasets_LLGF/LLGF_cora_train_pos.npy')

