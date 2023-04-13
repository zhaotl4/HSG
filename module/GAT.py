import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.GATStackLayer import MultiHeadLayer
from module.GATLayer import PositionwiseFeedForward, WSGATLayer, SWGATLayer


class WSWGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, atten_dropout,
                 ffn_inner_hidden_size, ffn_dropout, feat_embed_size,
                 layer_type) -> None:
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == "W2S":
            self.layer = MultiHeadLayer(in_dim,
                                        int(out_dim / num_heads),
                                        num_heads,
                                        atten_dropout,
                                        feat_embed_size,
                                        layer=WSGATLayer)
        elif self.layer_type == "S2W":
            self.layer = MultiHeadLayer(in_dim,
                                        int(out_dim / num_heads),
                                        num_heads,
                                        atten_dropout,
                                        feat_embed_size,
                                        layer_type=SWGATLayer)
        # elif self.layer=="S2S":
        # self.layer = MultiHeadSGATLayer()
        else:
            raise NotImplementedError("GAT Layer Not Found.")

        self.ffn = PositionwiseFeedForward(out_dim,                 ffn_inner_hidden_size,
                                           ffn_dropout)

    def forward(self, g, w, s):
        if self.layer_type == "W2S":
            origin, neighbor = s, w
        elif self.layer_type == "S2W":
            origin, neighbor = w, s
        else:
            origin, neighbor = None, None

        h = F.elu(self.layer(g, neighbor))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)

        return h
