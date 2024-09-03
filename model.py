import torch
from torch import nn
import torch.nn.functional as F
from utils import outputActivation
from attention import MultiheadAttention, EncoderBlock, TransformerEncoder
import matplotlib.pyplot as plt

    '''
+------------------------------------------------------------------------------------------------------------+
|                                               TrajPred Model                                               |
|                                                                                                            |
|  +------------------------------------+   +-------------------------+   +---------------------+            |
|  |       Temporal Convolution         |-->|   Neighbor GRU Encoder   |-->|   Dynamics Encoding  |            |
|  |                                    |   |                         |   |                     |            |
|  |  +-----------------------------+   |   | +-------------------+   |   | +-----------------+ |            |
|  |  |     Temporal Conv1D          |   |   | |        GRU        |   |   | |   Linear Layer  | |            |
|  |  +-----------------------------+   |   | +-------------------+   |   | +-----------------+ |            |
|  +------------------------------------+   +-------------------------+   +---------------------+            |
|                                                                                                            |
|  +------------------------------------+   +-------------------------+   +-------------------------------+ |
|  |  Social Context Convolution        |-->| Fully Convolutional Network (Fusion) |                        | |
|  |                                    |   |      (If enabled)                   |                        | |
|  |  +-----------------------------+   |   |                                     |                        | |
|  |  |  Conv2D + MaxPool2D          |   |   | +-----------------------------+    |                        | |
|  |  +-----------------------------+   |   | | Conv2D + ConvTranspose2D + BN|    |                        | |
|  +------------------------------------+   | +-----------------------------+    |                        | |
|                                           +------------------------------------+                        | |
|                                                                                                            |
|  +--------------------------+   +-------------------------+   +-------------------+   +------------------+ |
|  |     LSTM Decoder          |-->|  Transformer Encoder     |-->| Self-Attention     |-->| Output Prediction| |
|  |                           |   |                         |   | Module             |   |   Layers         | |
|  |  +---------------------+  |   |  +-------------------+  |   | +----------------+ |   | +-------------+ | |
|  |  |        LSTM          |  |   |  | Multi-Head       |  |   | | Multi-Head      | |   | |  Linear     | | |
|  |  +---------------------+  |   |  |  Attention        |  |   | |  Attention      | |   | |  Layers     | | |
|  +--------------------------+  |   |  +-------------------+  |   | +----------------+ |   | +-------------+ | |
|                                 |   |  +-------------------+  |   | +----------------+ |   | +-------------+ | |
|                                 |   |  | Feedforward Layers|  |   | |                | |   |                | |
|                                 |   |  +-------------------+  |   | |                | |   |                | |
|                                 +--------------------------------+---------------------+-------------------+ |
|                                                                                                            |
|                                                         +-----------------------------+                   |
|                                                         |       Future Prediction      |                   |
|                                                         |                               |                   |
|                                                         |  +------------------------+  |                   |
|                                                         |  |   Decode + Output      |  |                   |
|                                                         |  |    Activation          |  |                   |
|                                                         |  +------------------------+  |                   |
|                                                         +-----------------------------+                   |
+------------------------------------------------------------------------------------------------------------+

'''  

class TrajPred(nn.Module):
    def __init__(self, args):
        super(TrajPred, self).__init__()
        
         # Initialize out_length
        self.out_length = getattr(args, 'out_length', 10)  # Default to 10 if not provided

        # Save arguments for later use
        self.args = args
        self.use_cuda = args.use_cuda
        self.train_output_flag = args.train_output_flag
        self.use_planning = args.use_planning
        self.use_fusion = args.use_fusion
        
        # Define the number of lateral and longitudinal maneuver classes
        self.num_lat_classes = args.num_lat_classes
        self.num_lon_classes = args.num_lon_classes
        
        # Set the sizes for various network layers
        self.temporal_embedding_size = args.temporal_embedding_size
        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size
        self.soc_conv_depth = args.soc_conv_depth
        self.soc_conv2_depth = args.soc_conv2_depth
        self.dynamics_encoding_size = args.dynamics_encoding_size
        self.social_context_size = args.social_context_size
        
        # Set sizes for target and fusion encoding
        self.targ_enc_size = self.social_context_size + self.dynamics_encoding_size
        self.fuse_enc_size = args.fuse_enc_size
        
        # Define sizes for convolutional layers in the fusion network
        self.fuse_conv1_size = 2 * self.fuse_enc_size
        self.fuse_conv2_size = 4 * self.fuse_enc_size
        
        # Interactive Attention module flag
        self.IA_module = args.IA_module
        
        # Define activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Temporal Convolution to capture temporal consistency
        self.temporalConv = nn.Conv1d(in_channels=2, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
        
        # GRU for encoding the status of neighboring vehicles
        self.nbh_GRU = nn.GRU(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)
        
        # Linear layer to encode dynamics into a fixed size
        self.dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)
        
        # Convolutional layers for encoding social context
        self.nbrs_conv_social = nn.Sequential(
            nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3), 
            self.leaky_relu,
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
            self.leaky_relu
        )
        
        # Max pooling layer after merging social context
        self.pool_after_merge = nn.MaxPool2d((2, 1), padding=(1, 0))
        
        # old version
        # Fully Convolutional Network for fusion if fusion is enabled
        if self.use_fusion:
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
            
        # More sophisticated implementaion (version 1.0.1)

        # Define the dimensions for the decoder LSTM and output layers
        self.output_dim = self.num_lat_classes + self.num_lon_classes
        self.hidden_dim = self.targ_enc_size + self.fuse_enc_size
        self.op_lat = nn.Linear(self.hidden_dim, self.num_lat_classes)
        self.op_lon = nn.Linear(self.hidden_dim, self.num_lon_classes)
        self.dropout = nn.Dropout(p=0.1)
        
        
        # LSTM for decoding with self-attention
        max_length = self.num_lat_classes + self.num_lon_classes
        hidden_size_LSTM_dec = self.targ_enc_size + self.fuse_enc_size + max_length
        self.dec_lstm = nn.LSTM(hidden_size_LSTM_dec, self.decoder_size, bidirectional=False)
        
        # Output layer with Gaussian distribution output
        self.op_gauss_dim = 5
        self.op = nn.Linear(self.decoder_size, self.op_gauss_dim)
        
        # Self-attention module with multi-head attention
        num_heads = 2  # Set number of heads based on the dataset
        dropout = 0.5  # Dropout probability
        self.self_attn = MultiheadAttention(self.decoder_size, self.decoder_size, num_heads)
        self.dim_feedforward = self.hidden_dim
        
        # Define output networks
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
        
        # Layer normalization and dropout layers
        self.norm1 = nn.LayerNorm(self.decoder_size)
        self.norm2 = nn.LayerNorm(self.decoder_size)
        self.dropout = nn.Dropout(0.0)
        
        # Encoder block with multi-head attention for encoding
        self.op_net_3 = EncoderBlock(self.decoder_size, num_heads, self.dim_feedforward, dropout=dropout)
        
        # Transformer encoder for the decoder
        self.transformer = TransformerEncoder(
            num_layers=1,
            input_dim=self.decoder_size,
            dim_feedforward=self.dim_feedforward,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Redefine LSTM for the decoder based on IA_module flag
        if self.IA_module:
            self.dec_lstm = nn.LSTM(self.targ_enc_size + self.fuse_enc_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size, bidirectional=True)
        else:
            self.dec_lstm = nn.LSTM(self.targ_enc_size + self.fuse_enc_size, self.decoder_size, bidirectional=True)

    def forward(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc):
        """
        Forward pass for trajectory prediction model.
        """
        # Apply temporal convolution and activation to target history
        dyn_enc = self.leaky_relu(self.temporalConv(targsHist.permute(1, 2, 0)))

        # Encode dynamics with GRU and linear layer
        _, dyn_enc = self.nbh_GRU(dyn_enc.permute(2, 0, 1))
        dyn_enc = self.leaky_relu(self.dyn_emb(dyn_enc.view(dyn_enc.shape[1], dyn_enc.shape[2])))

        # Apply temporal convolution and activation to neighboring vehicle history
        nbrs_enc = self.leaky_relu(self.temporalConv(nbsHist.permute(1, 2, 0)))
        _, nbrs_enc = self.nbh_GRU(nbrs_enc.permute(2, 0, 1))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        
        # Initialize a grid for masked neighboring vehicles and apply the mask
        nbrs_grid = torch.zeros_like(nbsMask).float()
        nbrs_grid = nbrs_grid.masked_scatter_(nbsMask, nbrs_enc)
        nbrs_grid = nbrs_grid.permute(0, 3, 2, 1)
        nbrs_grid = self.nbrs_conv_social(nbrs_grid)
        
        # Pool after merging the grid
        merge_grid = self.pool_after_merge(nbrs_grid)
        social_context = merge_grid.view(-1, self.social_context_size)
        
        # Concatenate social context and dynamics encoding
        target_enc = torch.cat((social_context, dyn_enc), 1)
        
        # Initialize a grid for masked target encoding and apply the mask
        target_grid = torch.zeros_like(targsEncMask).float()
        target_grid = target_grid.masked_scatter_(targsEncMask, target_enc)
        
        # Apply fully convolutional layers if fusion is enabled
        if self.use_fusion:
            fuse_conv1 = self.relu(self.fcn_conv1(target_grid.permute(0, 3, 2, 1)))
            fuse_conv1 = self.bn1(fuse_conv1)
            fuse_conv1 = self.fcn_pool1(fuse_conv1)
            fuse_conv2 = self.relu(self.fcn_conv2(fuse_conv1))
            fuse_conv2 = self.bn2(fuse_conv2)
            fuse_conv2 = self.fcn_pool2(fuse_conv2)
            fuse_trans1 = self.relu(self.fcn_convTrans1(fuse_conv2))
            fuse_trans1 = self.back_bn1(fuse_trans1 + fuse_conv1)
            fuse_trans2 = self.relu(self.fcn_convTrans2(fuse_trans1))
            fuse_trans2 = self.back_bn2(fuse_trans2)
            
            # Apply mask to the fused grid and extract target locations
            fuse_grid_mask = targsEncMask[:, :, :, 0:self.fuse_enc_size]
            fuse_grid = torch.zeros_like(fuse_grid_mask).float()
            fuse_grid = fuse_grid.masked_scatter_(fuse_grid_mask, fuse_trans2.permute(0, 3, 2, 1))
            
            # Integrate target encoding and fused grid
            enc_rows_mark = targsEncMask[:, :, :, 0].view(-1)
            enc_rows = [i for i in range(len(enc_rows_mark)) if enc_rows_mark[i]]
            enc = torch.cat([target_grid, fuse_grid], dim=3)
            enc = enc.view(-1, self.fuse_enc_size + self.targ_enc_size)
            enc = enc[enc_rows, :]
        else:
            enc = target_enc
        
        # Decode based on the IA_module flag
        if self.IA_module:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))
            
            if self.train_output_flag:
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
        
    def decode(self, enc):
        """
        Decode the encoded input to predict future trajectories and visualize attention maps.
        """
        enc = enc.repeat(self.out_length, 1, 1)
    
        # Pass through LSTM
        h_dec, _ = self.dec_lstm(enc)
        # Combine forward and backward LSTM outputs if bidirectional
        if self.dec_lstm.bidirectional:
            h_dec = h_dec[:, :, :self.decoder_size] + h_dec[:, :, self.decoder_size:]
  
        # Permute to match expected input shape for the transformer
        h_dec = h_dec.permute(1, 0, 2)
        # Apply transformer encoder
        fut_pred = self.transformer(h_dec)
        # Retrieve attention maps
        attention_maps = self.transformer.get_attention_maps(h_dec)
        # Permute and apply final activation function
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
    
        return fut_pred

    
