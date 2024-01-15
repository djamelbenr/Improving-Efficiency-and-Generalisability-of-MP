#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Attention function 

"""
Created on Jan Thu 11 2024
@author: Djamel Eddine Benrachou, Sebastien Glaser, Andry...
"""
import torch 
from torch import nn 
from torchgviz import make_dot
from torch.distributions import Normal, OneHotCategorical
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt 

"""
Attention required modules.
1.dot_scaled_product
2.expand_mask (this is optional --)
3.encoder-block
4.multi-head attention encoder
"""
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


#-- masked attention can be added here.
def expand_mask(mask):
    assert mask.ndim >=2, 
      ##Mask must be at least 2-dimensional with seq_length x seq_length
    if mask.ndim==3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


def EncoderBlock(nn.Module):

    def __init__(self, input_dim, input_dim, num_heads)
    
    #Inputs: 
    #    input_dim - Dimensionality of the input
    #    num_heads - Number of heads to use in the attention block
    #    dim_feedforward - Dimensionality of the hidden layer in the MLP
    #    dropout - Dropout probability to use in the dropout layers
    
    super().__init__()
    
    #-- Attention layer 
    self.self_attn = nn.Sequential(
        nn.Linear(input_dim, dim_feedforward)
        nn.Dropout(dropout)
        nn.ReLU(inplace=True)
        nn.Linear(dim_feedforward, input_dim)
    )
# Layers to apply in between the main layers
#-- normalisation layers
    self.norm1=nn.LayerNorm(input_dim)
    self.norm2=nn.LayerNorm(input_dim)
    self.droptout=nn.Dropout(dropout)
## -- forward pass
def forward(self, x, mask=None):
    # Attention part
    attn_out = self.self_attn(x, mask=mask)
    x = x + self.dropout(attn_out)
    x = self.norm1(x)

    # MLP part
    linear_out = self.linear_net(x)
    x = x+ self.dropout(linear_out) 
    x = self.norm2(x)
    return x

## -- end. TransformerEndcoder

class TransforerEncoder(nn.Module):

    def __ini__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
        
    def forward(self,):
        for l in self.layers:
            x=l(x, mask=mask)        
        return x
        
    ## Optional: attention map visu
    def get_attention_maps():
        # -- visu the attention map
        # init 
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x=l(x)
        return attention_maps


## -- PositionalEncoding (optional)
class PositionalEncoding(nn.Module):
    """
    PositionalEncoding--Ref: arXiv:1706.03762v7  [cs.CL]  2 Aug 2023
                            "Attention is All You Need", 2018
    """
    def __init__(self, d_model, max_len=5000): 
        #-- set 5000 in this example
        """
        Inputs: 
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        #-Create matrix of [SeqLen, HiddenDim] representing the positional encoding../
        #/..for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position=torch.arange(0,max_len, dtype=torch.float).unsqueeze(1) 
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0) / d_model)) 
        # shift positions
        pe[:, 0::2]=torch.sin(position * div_term)
        pe[:, 1::2]=torch.cos(position * div_term)
        pe=pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part
        # of the modules state.
        # Used for tensors that need to be on the same device (GPU/TPU/CPU) as the module.
        # persistent=False tells Pytroch to not add the buffer to the state dict (e.g., when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x=x+self.pe[:,:x.size(1)]
        return x
##---------------------------------#
#|      Additional Modules         |
##---------------------------------#
#.(1)- self-attention (Attention), (2).spatial-attention, (3).temporal-attention
class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale=1./math.sqrt(query_dim)

    def forward(self, query, keys, values):
        ## Query = [BxQ]
        ## Keys  = [TxBxK]
        ## Values = [TxBxV]
        ## Outputs=a:[TxB], lin_comb:[BxV]
        # Here we assume :: q_dim =  k_dim (dot product attention)
        query = query.unsqueeze(1)                 # [BxQ]   -> [Bx1xQ]
        keys  = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        # energy
        energy=torch.bmm(query, keys)
        energy=F.softmax(energy.mul_(self.scale), dim=2)# scale, noramalise
        #get the values::
        values=values.transpose(0,1)
        linear_combination=torch.bmm(energy, values).squeeze(1) 
        # [Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination

#(2).spatial-attention
class SpatialAttn(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttn, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, H, W = l.size()
        c = self.op(l+g) # (batch_size,1,H,W)
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # (batch_size,C)
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,H,W), 


#(3).temporal-attention
class TemporalAttn(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        self.hidden_size= hidden_size 
        self.fc1=nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2=nn.Linear(self.hidden_size*2,self.hidden_size, bias=False)

    def forward(self, hidden_states):
        #(batch_size, time_steps, hidden_size)
        score_first_part=self.fc1(hidden_states)#(batch_size, hidden_size)
        h_t=hidden_states[:,-1,:]# (batch_size,time_steps)
        score=torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights=F.softmax(score, dim=1)#(batch_size,hidden_size)
        context_vector= torch.bmm(hidden_states.permute(0,2,1), attention_weights.unsqueeze(2)).squeeze(2)#
        pre_activation= torch.car((context_vector, h_t),dim=1)
        attention_vector=self.fc2(pre_activation)
        attention_vector=torch.tanh(attention_vector)
        
        return attention_vector, attention_weights

#--projection_block
class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features,
            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(x)
    
#-- Plot Attention Maps --
def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
    fig.subplots_adjust(hspace=0.5)
    plt.show()







    

    
        



    
    





  




