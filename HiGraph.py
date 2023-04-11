import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table


class HSumGraph(nn.Module):
    def __init__(self, hps, word_emb) -> None:
        super().__init__()
        self.hps = hps
        self.word_emb = word_emb
        self.word_emb_dim = hps.word_emb_dim
        self.n_iter = hps.n_iter

        # sent node feature
        self.init_sent_param()
        self.TFembed = nn.Embedding(10, hps.feat_embed_size)
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2,
                                        hps.hidden_size,
                                        bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        self.word2sent = WSWGAT(
            indim=embed_size,
            out_dim=hps.hidden_size,
            num_heads=hps.n_head,
            attn_drop_out=hps.atten_drop_prob,
            ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
            ffn_drop_out=hps.ffn_dropout_porb,
            feat_embed_size=hps.feat_embed_size,
            layerType="W2S")

        # sent -> word
        self.word2sent = WSWGAT(
            indim=hps.hidden_size,
            out_dim=embed_size,
            num_heads=6,
            attn_drop_out=hps.atten_drop_prob,
            ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
            ffn_drop_out=hps.ffn_dropout_porb,
            feat_embed_size=hps.feat_embed_size,
            layerType="S2W")

        # node classification
        self.n_feature = hps.hidden_size
        self.wh = nn.Linear(self.n_feature, 2)

    def init_sent_param(self):
        self.sent_pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.hps.doc_max_timesteps + 1,
                                        self.word_emb_dim,
                                        padding_idx=0),
            freeze=True)
        # 映射函数 word dim(300) -> node feature dim(128)
        self.cnn_proj = nn.Linear(self.word_emb_dim, self.hps.n_feature_size)
        self.lstm_hidden_state = self.hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.word_emb_dim,
                            self.lstm_hidden_state,
                            num_layers=self.hps.lstm_layers,
                            dropout=0.1,
                            batch_first=True,
                            bidirectional=self.hps.bidirectional)
        if self.hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2,
                                       self.hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state,
                                       self.hps.n_feature_size)

        self.ngram_enc = sentEncoder(self.hps, self.word_emb)
