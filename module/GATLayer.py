import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1) -> None:
        super().__init__()
        self.w_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.w_2 = nn.Conv1d(hidden_dim, in_dim, 1)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x))

        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)

        assert not torch.any(torch.isnan(x))

        return output


class WSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data['tfidfembed'])
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)
        # eq 6
        wa = F.leaky_relu(self.attn_fc(z2))

        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        #eq 2-3
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'sh': h}

    def forward(self, g, h):
        word_node_id = g.filter_nodes(lambda nodes: nodes.data['unit'] == 0)
        sent_node_id = g.filter_ndoes(lambda nodes: nodes.data['unit'] == 1)
        word_sent_edge_id = g.filter_edges(
            lambda edges: (edges.src['unit'] == 0) & edges.dst['unit'] == 1)

        # h is the neighbor
        z = self.fc(h)
        g.nodes[word_node_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=word_sent_edge_id)
        g.pull(sent_node_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')

        return h[sent_node_id]


class SWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data['tfidfembed'])
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)
        wa = F.leaky_relu(self.attn_fc(z2))

        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) &
                                   (edges.dst["unit"] == 0))
        z = self.fc(h)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=swedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]