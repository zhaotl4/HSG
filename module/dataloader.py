import re
import os
from nltk.corpus import stopwords

import glob
import copy
import random
import time
import json
import pickle
import nltk
import networkx as nx
import collections
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle

import torch
import torch.utils.data
import torch.nn.functional as F

import dgl
from dgl.data.utils import save_graphs, load_graphs

FILTERWORD = stopwords.words('english')
punctuations = [
    ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$',
    '%', '\'\'', '\'', '`', '``', '-', '--', '|', '\/'
]
FILTERWORD.extend(punctuations)


def readJson(datapath):
    data = []
    with open(datapath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # json.loads()将str类型的数据转换为dict类型
            item = json.loads(line)
            data.append(item)
    f.close()

    return data


def readText(datapath):
    data = []
    with open(datapath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append[line.strip()]
    f.close()

    return data


class Example(object):
    def __init__(self, text, summary, label, max_len, vocab):
        self.sent_max_len = max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        self.origin_text = text
        self.origin_summary = summary

        for sent in self.origin_text:
            word_in_sent = sent.split()
            self.enc_sent_len.append(len(word_in_sent))
            line = []
            for word in word_in_sent:
                word_id = vocab.word2id[word.lower()]
                line.append(word_id)
            self.enc_sent_input.append(line)

        # PAD the sent to max(len(word_in_sent))
        self.pad_input_sent(vocab.word2id('[PAD]'))

        self.label = label
        label_shape = (len(self.origin_text), len(label))  # [N, len(label)]
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(
                label
            ), np.arange(
                len(label)
            )] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

    def pad_input_sent(self, pad_id):
        # sent is a list, self.enc_sent_input is a list(list)
        for sent in self.enc_sent_input:
            sent_len = len(sent)
            sent_copy = sent.copy()  # make a copy
            if sent_len > self.sent_max_len:
                self.enc_sent_input_pad.append(sent_copy[:self.sent_max_len])
            else:
                diff_len = self.sent_max_len - sent_len
                pad_list = [pad_id] * diff_len
                pad_sent = sent_copy.extend(pad_list)
                self.enc_sent_input_pad.append(pad_sent)


class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len,
                 filter_word_path, w2s_path):
        """ Initializes the ExampleSet with the path of data

        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        """
        super().__init__()
        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps
        '''
        example_list is a list and each item is a dict

        {'text': ['maxi lopez appeared to be sending a message to former team-mate mauro icardi and ex-wife wanda nara as pictures emerged of him on holiday in miami with a new girlfriend .', ... ,"unfazed : icardi has n't let the attention of his relationship with nara effect his performances", 'solitary : lopez only managed to score one league goal this season'], 'label': [0, 1, 12]}
        '''
        self.example_list = readJson(data_path)
        self.size = len(self.example_list)

        self.filter_words = FILTERWORD
        self.filter_ids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filter_ids.append(
            vocab.word2id("[PAD]"))  # keep "[UNK]" but remove "[PAD]"

        all_words = readText(filter_word_path)
        cnt = 0
        for word in all_words:
            if vocab.word2id(word) != vocab.word2id('[UNK]'):
                self.filter_words.append(word)
                self.filter_ids.append(vocab.word2id(word))
                cnt += 1
            if cnt > 5000:
                break

        self.w2s_tfidf = readJson(w2s_path)

    def __getitem__(self, index):
        example = self.get_example(index)
        input_pad = example.enc_sent_input_pad  # sent after pad
        input_w2s_tfidf = self.w2s_tfidf[index]
        label_pad = self.pad_label(example.label_matrix)
        G = self.create_graph(input_pad, label_pad, input_w2s_tfidf)
        return G, index

    def __len__(self):
        return self.size

    def get_example(self, idx):
        item = self.example_list[idx]
        item['summary'] = item.setdefault('summary', [])
        example = Example(item['text'], item['summary'], item['label'],
                          self.sent_max_len, self.vocab)
        return example

    def pad_label(self, label_matrix):
        matrix_copy = label_matrix.copy()
        pad_label = matrix_copy[:self.doc_max_timesteps, :self.
                                doc_max_timesteps]
        N, m = pad_label.shape
        if m < self.doc_max_timesteps:
            pad_item = np.zeros((N, self.doc_max_timesteps - m))
            pad_label = np.hstack(pad_label, pad_item)

        return pad_label

    def add_word_node(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        node_cnt = 0
        for sent in inputid:
            for word_id in sent:
                if word_id not in wid2nid.keys(
                ) and word_id not in self.filter_ids:
                    wid2nid[word_id] = node_cnt
                    nid2wid[node_cnt] = word_id
                    node_cnt += 1

        G.add_nodes(node_cnt)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata['dtype'] = torch.zeros(node_cnt)
        G.ndata['unit'] = torch.zeros(node_cnt)
        G.ndata['id'] = torch.LongTensor(list(nid2wid.values()))

        return G

    def create_graph(self, input_pad, label, w2s_tfidf):

        G = dgl.DGLGraph()

        wid2nid, nid2wid = self.add_word_node(G, input_pad)
        word_cnt = len(nid2wid)

        sent_cnt = len(input_pad)
        G.add_nodes(sent_cnt)
        G.ndata['dtype'][word_cnt:] = torch.ones(sent_cnt)
        G.ndata['unit'][word_cnt:] = torch.ones(sent_cnt)
        sent_ids = [i + word_cnt for i in range(sent_cnt)]

        G.set_e_initializer(dgl.init.zero_initializer)

        for i in range(sent_cnt):
            c = Counter(input_pad[i])
            sent_node = sent_ids[i]
            tfidf_matrix = w2s_tfidf[str(i)]

            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(
                        wid) in tfidf_matrix.keys():
                    word_node = wid2nid[wid]
                    tfidf_value = tfidf_matrix[self.vocab.id2word(wid)]
                    tfidf_value = np.round(tfidf_value * 9)  # do not know why
                    G.add_edges(word_node,
                                sent_node,
                                data={
                                    "tffrac": torch.LongTensor([tfidf_value]),
                                    "dtype": torch.tensor([0])
                                })
                    G.add_edges(sent_node,
                                word_node,
                                data={
                                    "tffrac": torch.LongTensor([tfidf_value]),
                                    "dtype": torch.tensor([0])
                                })

            # leave or comment these two lines
            G.add_edges(sent_node,
                        sent_ids,
                        data={"dtype": torch.ones(sent_cnt)})
            G.add_edges(sent_ids,
                        sent_node,
                        data={"dtype": torch.ones(sent_cnt)})

        G.nodes[sent_ids].data["words"] = torch.LongTensor(input_pad)
        G.nodes[sent_ids].data["position"] = torch.arange(1,
                                                          sent_cnt + 1).view(
                                                              -1, 1).long()
        G.nodes[sent_ids].data["label"] = torch.LongTensor(label)


def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return: 
    '''
    graphs, index = map(list, zip(*samples))
    graph_len = [
        len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1))
        for g in graphs
    ]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len),
                                          dim=0,
                                          descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]
