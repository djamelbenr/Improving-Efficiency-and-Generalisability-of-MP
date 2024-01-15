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
    """
    Inputs: 
        input_dim - Dimensionality of the input
        num_heads - Number of heads to use in the attention block
        dim_feedforward - Dimensionality of the hidden layer in the MLP
        dropout - Dropout probability to use in the dropout layers
    """
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


    def forward():
        
        return x
        
    ## Optional: attention map visu
    def get_attention_maps():
        # ini 
        attention_maps = []
        for l in self.layers: 


        return attention_maps




    
    





  




