#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Attention function 

"""
Created on Jan Thu 11 2024
@author: Djamel Eddine Benrachou, et al
"""

import torch 
from torch import nn 
from torchgviz import make_dot
from torch.distributions import Normal, OneHotCategorical
import torch.nn.functional as F
import numpy as np 
import math 
import matplotlib.pyplot as plt 




