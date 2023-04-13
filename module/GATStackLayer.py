import torch
import torch.nn as nn

class MultiHeadLayer(nn.Module):
    def __init__(self,in_dim,out_dim,num_heads,atten_dropout,feat_embed_size,layer,merge='cat') -> None:
        super().__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(layer(in_dim,out_dim,feat_embed_size))
        
        self.merge = merge
        self.dropout = nn.Dropout(atten_dropout)

    def forward(self,g,h):
        head_outs = [atten_head(g,self.dropout(h)) for atten_head in self.heads]

        if self.merge=='cat':
            result = torch.cat(head_outs,dim=1)
        else:
            result = torch.mean(torch.stack(head_outs))
        
        return result
    