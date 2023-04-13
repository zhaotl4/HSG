from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from module.PositionEmbedding import get_sinusoid_encoding_table

WORD_PAD = "[PAD]"


class sentEncoder(nn.Module):
    def __init__(self, hps, embed) -> None:
        super(sentEncoder).__init__()
        self.hps = hps
        self.sent_max_len = hps.sent_max_len
        embed_size = hps.word_emb_dim

        input_channels = 1
        outout_channels = 50
        min_kernel_size = 2
        max_kernel_size = 7
        width = embed_size

        self.embed = embed

        # poisition embedding
        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.sent_max_len + 1,
                                        embed_size,
                                        padding_idx=0),
            freeze=True)

        # cnn
        self.convs = nn.ModuleList([
            nn.Conv2d(input_channels,
                      outout_channels,
                      kernel_size=(height, width))
            for height in range(min_kernel_size, max_kernel_size + 1)
        ])

        # init the weight of cnn
        for conv in self.convs:
            init_weight_value = 6.0
            init.xavier_normal_(conv.weight.data,gain=np.sqrt(init_weight_value))
    
    def forward(self,input):
        # input is a batch of example object [s_nodes, seq_len]
        input_sent_len = ((input!=0).sum(dim=1)).int() # [s_nodes, 1]

        enc_embed_input = self.embed(input)

        sent_pos_list = []
        for sent_len in input_sent_len:
            sent_pos = list(range(1,min(self.sent_max_len,sent_len)))
            sent_pos.extend([0]* int(self.sent_max_len-sent_len))
            sent_pos_list.append(sent_pos)

        input_pos = torch.Tensor(sent_pos_list).long()

        if self.hps.cuda:
            input_pos = input_pos.cuda()

        enc_pos_emb_input = self.position_embedding(input_pos.long())
        enc_conv_input = enc_pos_emb_input + enc_embed_input
        enc_conv_input = enc_conv_input.unsqueeze(1)
        enc_conv_output = [F.relu(conv(enc_conv_input)).squeeze(3) for conv in self.convs]
        enc_maxpool_output = [F.max_pool1d(x,x.size(2)).squeeze(2) for x in enc_conv_output]
        sent_embedding = torch.cat(enc_maxpool_output,1)

        return sent_embedding
