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
2.
3.
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
  




