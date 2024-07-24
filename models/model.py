#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

## Attention function 

"""
Created on Jan Thu 24 2024
@author: Djamel Eddine Benrachou et al.
"""

import torch 
from torch import nn
from utils import outputActivation
from torchviz import make_dot 
from torch.distributions import Normal, OneHotCategorical 
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt 

# Function to compute scaled dot-product attention
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]  # Dimension of the key
    attn_logits = torch.matmul(q, k.transpose(-2, -1))  # Compute the attention logits
    attn_logits = attn_logits / math.sqrt(d_k)  # Scale the attention logits
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)  # Apply mask
    attention = F.softmax(attn_logits, dim=-1)  # Compute softmax
    values = torch.matmul(attention, v)  # Compute the attention-weighted values
    return values, attention

# Class for a projection block using a 1x1 convolution
class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                            kernel_size=1, padding=0, bias=False)
        
    def forward(self, x):
        return self.op(x)

# Helper function to expand mask dimensions to 4D
def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

# Class for multi-head attention
class multiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)  # Linear layer to project input to Q, K, V
        self.o_proj   = nn.Linear(embed_dim, embed_dim)  # Linear layer to project output

        self._reset_parameters()  # Initialize parameters
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)  # Compute Q, K, V
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # Reshape to [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)  # Separate Q, K, V

        values, attention = scaled_dot_product(q, k, v, mask=mask)  # Compute scaled dot-product attention
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)  # Project output

        if return_attention:
            return o, attention
        else:
            return o

# Class for transformer encoder block
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout):
        super().__init__()

        self.self_attn = multiheadAttention(input_dim, input_dim, num_heads)  # Multi-head attention

        # Two-layer feedforward network
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)  # Apply self-attention
        x = x + self.dropout(attn_out)  # Add & normalize
        x = self.norm1(x)
        linear_out = self.linear_net(x)  # Apply feedforward network
        x = x + self.dropout(linear_out)  # Add & normalize
        x = self.norm2(x)
        return x

# Class for transformer encoder with multiple layers
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x
            
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

# Class for spatial attention block (CBAM)
class SpatialAttn(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttn, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1,
                            kernel_size=1, padding=0, bias=False)
                                        
    def forward(self, l, g):
        N, C, H, W = l.size()
        c = self.op(l + g)
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, H, W)
        else: 
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(1), 1)
        if self.normalize_attn:
            g = g.view(N, C, -1).sum(dim=2)  # (batch_size, C)
        else: 
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)
        return c.view(N, 1, H, W), g 

# Class for temporal attention block
class TemporalAttn(nn.Module): 
    def __init__(self, hidden_size):
        super(TemporalAttn, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        score_first_part = self.fc1(hidden_states)  # (batch_size, time_steps, hidden_size)
        h_t = hidden_states[:, -1, :]  # (batch_size, hidden_size)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)  # (batch_size, time_steps)
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.bmm(hidden_states.permute(0, 2, 1), attention_weights.unsqueeze(2)).squeeze(2)  # (batch_size, hidden_size)
        pre_activation = torch.cat((context_vector, h_t), dim=1)  # (batch_size, hidden_size * 2)
        attention_vector = self.fc2(pre_activation)  # (batch_size, hidden_size)
        attention_vector = torch.tanh(attention_vector)

        return attention_vector, attention_weights

# Function to plot attention maps
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
      
    
class TrajPred(nn.Module):
    def __init__(self, args):
        super(TrajPred, self).__init__()
        # Store the arguments
        self.args = args
        self.use_cuda = args.use_cuda
        
        # Flag for output mode:
        # Train-mode: concatenate with "True" manoeuvre labels
        # Test-mode: concatenate with predicted manoeuvres with max probability
        self.train_output_flag = args.train_output_flag
        
        # Fusion information conditions
        self.use_fusion = args.use_fusion
        
        # Setting up I/O parameters
        self.grid_size = args.grid_size
        self.in_length = args.in_length
        self.out_length = args.out_length
        # <------------------------------>#<---------------------------<----->|
        #    n_Hist                       .   n_Fut                    |      | Past/Fut
        # +-------------------------------+---------------------------><------|
        # <------------------------------>.<--------------------------><----->| 
        # lat_enc         [405, 3]        .                            |      | LK, RLC, LLC
        # lon_enc         [405, 2]        .                            |maneuv| normal, braking  
        # <------------------------------>.<--------------------------><----->| Tensors 
        # Number of maneuver classes
        self.num_lat_classes = args.num_lat_classes
        self.num_lon_classes = args.num_lon_classes
        
        # Network layer sizes
        self.temporal_embedding_size = args.temporal_embedding_size
        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size
        self.soc_conv_depth = args.soc_conv_depth
        self.soc_conv2_depth = args.soc_conv2_depth
        self.dynamics_encoding_size = args.dynamics_encoding_size
        self.social_context_size = args.social_context_size
        
        # Target and Fusion encoding sizes with social context
        self.targ_enc_size = self.social_context_size + self.dynamics_encoding_size
        self.fuse_enc_size = args.fuse_enc_size
        self.fuse_conv1_size = 2 * self.fuse_enc_size
        self.fuse_conv2_size = 4 * self.fuse_enc_size
        
        # Intermediate activation module
        self.IA_module = args.IA_module
        
        # Bottleneck dimension
        self.bottleneck_dim = 1024
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Bidirectional flag for LSTM
        bidirectional = True
        ## Helper!
        # <------------------------------>#<---------------------------<----->|
        #    n_Hist                       .   n_Fut                    |      | Past/Fut
        # +-------------------------------+---------------------------><------|
        # |nbsHist|nbsHist| [16, nb_nbs, 2].                           |      |  
        # +-------------------------------+.                           |inputs| Tensors
        # |targsHist|      [16, nb_targs,2]. |nbsFut|[25, nb_targs, 2 ]|      | 
        # +-------------------------------+.                           |      | 
        # <------------------------------>#<--------------------------><----->| 
        # nbsMask         [405, 5, 25, 64].                            |      | 
        # targsEncMask    [64, 5, 25, 112].                            |Masks | Tensors 
        #                                 .targsFutMask    [25, 405, 2]|      | 
        # <------------------------------>.<--------------------------><----->| 
        # lat_enc         [405, 3]        .                            |      | LK, RLC, LLC
        # lon_enc         [405, 2]        .                            |maneuv| normal, braking 
        #                                 .                            |      | Tensors
        # <------------------------------>.<--------------------------><----->|
        # Temporal embedding on the encoded dynamics
        self.temporalConv = nn.Conv1d(in_channels=2, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
        
        # Encode the neighboring vehicles using GRU
        self.nbh_GRU = nn.GRU(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)
        
        # Encoded dynamics: dynamics encoding size
        self.dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)
        
        # Social context convolutional layers
        self.nbrs_conv_social = nn.Sequential(
            nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
            self.leaky_relu,
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, 3, 1),
            self.leaky_relu
        )
        
        # Max pooling after merging
        self.pool_after_merge = nn.MaxPool2d((2, 1), padding=(1, 0))
        
        # Target Fusion Module
        if self.use_fusion:
            # Fully connected layers for fusion
            self.fcn_conv1 = nn.Conv2d(self.targ_enc_size, self.fuse_conv1_size, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(self.fuse_conv1_size)
            self.fcn_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.fcn_conv2 = nn.Conv2d(self.fuse_conv1_size, self.fuse_conv2_size, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(self.fuse_conv2_size)
            self.fcn_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.fcn_convTrans1 = nn.ConvTranspose2d(self.fuse_conv2_size, self.fuse_conv1_size, kernel_size=3, stride=2, padding=1)
            self.back_bn1 = nn.BatchNorm2d(self.fuse_conv1_size)
            self.fcn_convTrans2 = nn.ConvTranspose2d(self.fuse_conv1_size, self.fuse_enc_size, kernel_size=3, stride=2, padding=1)
            self.back_bn2 = nn.BatchNorm2d(self.fuse_enc_size)
        else:
            self.fuse_enc_size = 0
        
        # Decoder level 1
        self.output_dim = self.num_lat_classes + self.num_lon_classes
        self.hidden_dim = self.targ_enc_size + self.fuse_enc_size
        
        # Longitudinal and lateral control
        self.op_lat = nn.Linear(self.hidden_dim, self.num_lat_classes)
        self.op_lon = nn.Linear(self.hidden_dim, self.num_lon_classes)
        self.dropout = nn.Dropout(p=0.1)
        
        # LSTM Decoder setup
        max_length = nn.num_lat_classes + self.num_lon_classes
        hidden_size_LSTM_dec = self.targ_enc_size + self.fuse_enc_size + max_length
        output_size_LSTM_dec = self.decoder_size
        self.dec_lstm = nn.LSTM(hidden_size_LSTM_dec, output_size_LSTM_dec, bidirectional=bidirectional)
        
        # Output Gaussian dimension
        self.op_gauss_dim = 5
        self.op = nn.Linear(self.decoder_size, self.op_gauss_dim)
        
        # Multihead attention parameters
        num_heads = 2
        dropout = 0.5
        self.self_attn = multiheadAttention(self.decoder_size, self.decoder_size, num_heads)
        self.dim_feedforward = self.hidden_dim
        
        # Multi-Layer Perceptrons (MLPs)
        self.op_net = nn.Sequential(
            nn.Linear(self.decoder_size, self.dim_feedforward),
            nn.Dropout(0.0),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_feedforward, self.output_dim)
        )
        self.op_net_2 = nn.Sequential(
            nn.Linear(self.decoder_size, self.dim_feedforward),
            nn.Dropout(0.0),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_feedforward, self.decoder_size)
        )
        self.norm1 = nn.LayerNorm(self.decoder_size)
        self.norm2 = nn.LayerNorm(self.decoder_size)
        self.dropout = nn.Dropout(0.0)
        
        # Encoder block and transformer setup
        self.op_net_3 = EncoderBlock(self.decoder_size, num_heads, self.dim_feedforward, dropout=dropout)
        self.transformer = TransformerEncoder(
            num_layers=1,
            input_dim=self.decoder_size,
            dim_feedforward=self.dim_feedforward,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # One-Hot encoding if IA_module is true
        if self.IA_module:
            self.dec_lstm = nn.LSTM(
                self.targ_enc_size + self.fuse_enc_size + self.num_lat_classes + self.num_lon_classes,
                self.decoder_size, bidirectional=True
            )
        else:
            self.dec_lstm = nn.LSTM(
                self.targ_enc_size + self.fuse_enc_size, self.decoder_size, bidirectional=True
            )
        
        # Redefine previously defined layers due to repeated code
        self.dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, nbsHist, nbsMask, targsHist, targsEncMask, lat_enc, lon_enc):
        # Apply temporal convolution on the target history
        dyn_enc = self.leaky_relu(self.temporalConv(targsHist.permute(1, 2, 0)))
        # Encode target history using GRU
        output, dyn_enc = self.nbh_GRU(dyn_enc.permute(2, 0, 1))
        dyn_enc = self.leaky_relu(self.dyn_emb(dyn_enc.view(dyn_enc.shape[1], dyn_enc.shape[2])))
        
        # Forward neighboring vehicles
        nbrs_enc = self.leaky_relu(self.temporalConv(nbsHist.permute(1, 2, 0)))
        output, nbrs_enc = self.nbh_GRU(nbrs_enc.permute(2, 0, 1))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        
        # Create a grid for neighboring vehicles
        nbrs_grid = torch.zeros_like(nbsMask).float()
        nbrs_grid = nbrs_grid.masked_scatter_(nbsMask, nbrs_enc)
        nbrs_grid = nbrs_grid.permute(0, 3, 2, 1)
        nbrs_grid = self.nbrs_conv_social(nbrs_grid)
        
        # Define input sizes for social context convolution
        self.input_size = self.encoder_size
        self.hidden_size_1 = self.soc_conv_depth
        self.hidden_size_2 = self.soc_conv2_depth
        
        # Merge the grid
        merge_grid = merge_grid.view(-1, self.social_context_size)
        
        # Concatenate social context (neighbors + ego) and dynamics encoding, place into the target encoding mask
        target_enc = torch.cat((social_context, dyn_enc), 1)
        target_grid = torch.zeros_like(targsEncMask).float()
        target_grid = target_grid.masked_scatter_(targsEncMask, target_enc)
        
        # Target Fusion
        if self.use_fusion:
            # Fully connected layers for fusion
            fuse_conv1 = self.relu(self.fcn_conv1(target_grid.permute(0, 3, 2, 1)))
            fuse_conv1 = self.bn1(fuse_conv1)
            fuse_conv1 = self.fcn_pool1(fuse_conv1)
            fuse_conv2 = self.relu(self.fcn_conv2(fuse_conv1))
            fuse_conv2 = self.bn2(fuse_conv2)
            fuse_conv2 = self.fcn_pool2(fuse_conv2)
            
            # Encode/Decode fusion layers
            fuse_trans1 = self.relu(self.fcn_convTrans1(fuse_conv2))
            fuse_trans = self.back_bn1(fuse_trans1 + fuse_conv1)
            fuse_trans2 = self.relu(self.fcn_convTrans2(fuse_trans1))
            fuse_trans2 = self.back_bn2(fuse_trans2)
            
            # Extract the location with targets
            fuse_grid_mask = targsEncMask[:, :, :, 0:self.fuse_enc_size]
            fuse_grid = torch.zeros_like(fuse_grid_mask).float()
            fuse_grid = fuse_grid.masked_scatter_(fuse_grid_mask, fuse_trans2.permute(0, 3, 2, 1))
            
            # Integrate everything together
            enc_rows_mark = targsEncMask[:, :, :, 0].view(-1)
            enc_rows = [i for i in range(len(enc_rows_mark)) if enc_rows_mark[i]]
            enc = torch.cat([target_grid, fuse_grid], dim=3)
            enc = enc.view(-1, self.fuse_enc_size + self.targ_enc_size)
            enc = enc[enc_rows, :]
        else:
            enc = target_enc
        
        if self.IA_module:
            # Predict lateral and longitudinal maneuvers
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))
            
            if self.train_output_flag:
                # Concatenate with true maneuver labels
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                fut_pred = self.decode(enc)
            else:
                fut_pred = []
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
            return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc)
            return fut_pred
    
    # Decode function to generate future predictions
    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, (h_n, c_n) = self.dec_lstm(enc)
        u_emb_batch = h_dec[:, :, :self.decoder_size] + h_dec[:, :, self.decoder_size:]
        fut_pred = self.transformer(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred
    
    

        
        
        
        
            
            
        
 
        
        

