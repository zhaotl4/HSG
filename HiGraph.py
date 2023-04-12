import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

from module.GAT import GAT, GAT_ffn
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
        self.sent2word = WSWGAT(
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

    def set_word_node_feature(self, graph):
        word_node_id = graph.filter_nodes(
            lambda nodes: nodes.data["unit"] == 0)
        word_sent_edge_id = graph.filter_edges(
            lambda edges: edges.data["dtype"] == 0)
        word_id = graph.nodes[word_node_id].data["id"]
        # get the word embedding
        w_emb = self.word_emb(word_id)
        graph.nodes[word_node_id].data['embed'] = w_emb
        edge_tfidf = graph.edges[word_sent_edge_id].data['tffrac']
        edge_tfidf_emb = self.TFembed(edge_tfidf)
        graph.edges[word_sent_edge_id].data['tfidfembed'] = edge_tfidf_emb

        return w_emb

    def set_sent_node_feature(self, graph):
        sent_node_id = graph.filter_nodes(
            lambda nodes: nodes.data['unit'] == 1)
        cnn_feature = self.sent_cnn_feature(graph, sent_node_id)
        feat, glen = get_sent_node_feat(graph, feat="sent_embedding")
        lstm_feature = self.sent_lstm_feature(feat, glen)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)

        return node_feature

    def sent_cnn_feature(self, graph, sent_node_id):
        ngram_feature = self.ngram_enc.forward(
            graph.nodes[sent_node_id].data['words'])
        graph.nodes[sent_node_id].data['sent_embedding'] = ngram_feature
        sent_node_pos = graph.nodes[sent_node_id].data['position'].view(-1)
        position_embedding = self.sent_pos_emb(sent_node_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)

        return cnn_feature

    def sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output,
                                                         batch_first=True)
        lstm_embedding = [
            unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))
        ]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))

        return lstm_feature

    def forward(self, graph):
        word_feature = self.set_word_node_feature(graph)
        sent_feature = self.n_feature_proj(self.set_sent_node_feature(graph))

        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self.n_iter):
            word_state = self.sent2word(graph, word_state, sent_state)
            sent_state = self.word2sent(graph, word_state, sent_state)

        result = self.wh(sent_state)

        return result


def get_sent_node_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        sent_node_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
        feature.append(g.nodes[sent_node_id].data[feat])
        glen.append(len(sent_node_id))

    return feature, glen
