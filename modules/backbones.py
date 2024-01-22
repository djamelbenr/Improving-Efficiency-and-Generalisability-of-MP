import torch
from torch import nn
from utils import outputActivation

# import attention module.
class TrajPred(nn.Module):
# <------------------------------>#<---------------------------<----->|
#    n_Hist                       .   n_Fut                    |      | Past/Fut
# +-------------------------------+---------------------------><------|
# |nbsHist|nbsHist| [16, nb_nbs, 2].                           |      |  
# +-------------------------------+.                           |inputs| Tensors
#|targsHist|       [16, nb_targs,2]. |nbsFut|[25, nb_targs, 2 ]|      | 
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

class TrajPred(nn.Module):
    # Class for trajectory prediction model.

    def __init__(self, args):
        # Initialize the trajectory prediction model.

        super(TrajPred, self).__init__()
        self.args = args
        self.use_cuda = args.use_cuda
        # Flag for output:
        # -- Train-mode : Concatenate with true maneuver label.
        # -- Test-mode  : Concatenate with the predicted maneuver with the maximal probability.
        self.train_output_flag = args.train_output_flag

        ## Planning and fusion information conditions
        self.use_planning = args.use_planning
        self.use_fusion = args.use_fusion

        # Setting up the I/O 
        self.grid_size = args.grid_size
        self.in_length = args.in_length
        self.out_length = args.out_length

        # Lateral and longitudinal classes.
        self.num_lat_classes = args.num_lat_classes
        self.num_lon_classes = args.num_lon_classes

        # Setup size of layers.
        self.temporal_embedding_size = args.temporal_embedding_size

        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size

        self.soc_conv_depth = args.soc_conv_depth
        self.soc_conv2_depth = args.soc_conv2_depth

        self.dynamics_encoding_size = args.dynamics_encoding_size
        self.social_context_size = args.social_context_size

        # Target encoder size.
        self.targ_enc_size = self.social_context_size + self.dynamics_encoding_size

        # Conv Nets for information fusion.
        self.fuse_enc_size = args.fuse_enc_size  # Info fusion.
        self.fuse_conv1_size = 2 * self.fuse_enc_size
        self.fuse_conv2_size = 4 * self.fuse_enc_size
        self.bidirectional = True

        # Activations and outputs.
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        ## Define network parameters
        ''' Convert traj to temporal embedding'''
        ''' Add temporal consistency on the encoded dynamic : '''
        # Time arguments.
        self.temporalConv = nn.Conv1d(in_channels=2, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)

        ''' Encode the input temporal embedding '''
        self.nbh_GRU = nn.GRU(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)
        self.nbh_bilstm = nn.LSTM(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1, bidirectional=True)  # biLSTM decoder

        ''' Encoded dynamic to dynamics_encoding_size'''
        self.dyn_embedding = nn.Embedding(self.encoder_size, self.dynamics_encoding_size)

        self.dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)
        ######################################################
        # The Linear layer should be replaced by an MDN layer.
        # replace dyn_emb
        # start MDN implementation.
        ######################################################
        ''' Convolutional Social Pooling on the planned vehicle and all neighbors' vehicles  '''
        self.nbrs_conv_social = nn.Sequential(
            nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
            self.leaky_relu,
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
            self.leaky_relu
        )
        self.pool_after_merge = nn.MaxPool2d((2, 1), padding=(1, 0))
        # Information fusion turned off - on if fusion info is == True
        ''' Target Fusion Module'''
        #######################################################
        # convolution fusion : original network.
        #######################################################
        # Implement StrideNet
        if self.use_fusion:
            """
            fuse_enc_size   = 112 
            fuse_conv1_size = 2 * fuse_enc_size
            fuse_conv2_size = 4 * fuse_enc_size
            """
            self.fcn_conv1 = nn.Conv2d(self.targ_enc_size, self.fuse_conv1_size, kernel_size=7, stride=1, padding=1)
            self.dropout1 = nn.Dropout(p=0.1)
            self.bn1 = nn.BatchNorm2d(self.fuse_conv1_size)
            self.fcn_conv2 = nn.Conv2d(self.fuse_conv1_size, self.fuse_conv2_size, kernel_size=3, stride=2, padding=1)
            self.dropout2 = nn.Dropout(p=0.1)
            self.bn2 = nn.BatchNorm2d(self.fuse_conv2_size)
            self.fcn_convTrans1 = nn.ConvTranspose2d(self.fuse_conv2_size, self.fuse_conv1_size, kernel_size=3, stride=2, padding=1)
            self.dropout3 = nn.Dropout(p=0.2)
            self.back_bn1 = nn.BatchNorm2d(self.fuse_conv1_size)
            self.fcn_convTrans2 = nn.ConvTranspose2d(self.fuse_conv1_size, self.fuse_enc_size, kernel_size=3, stride=2, padding=1)
            self.dropout4 = nn.Dropout(p=0.3)
            self.back_bn2 = nn.BatchNorm2d(self.fuse_enc_size)
        else:
            self.fuse_enc_size = 0

        ''' Decoder:  LSTM'''
        # MDN for sequential maneuver forms
        self.output_dim = self.num_lat_classes + self.num_lon_classes
        self.hidden_dim = self.targ_enc_size + self.fuse_enc_size
        self.embedding = nn.Embedding(self.output_dim, self.hidden_dim)

        self.op_lat = nn.Linear(self.hidden_dim, self.num_lat_classes)  # Output lateral maneuver.
        self.op_lon = nn.Linear(self.hidden_dim, self.num_lon_classes)  # Output longitudinal maneuver.

        self.dropout = nn.Dropout(p=0.1)

        self.IA_module = args['intention_module']
        input_size_LSTM_dec_intention = self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes
        hidden_size_LSTM_dec_intention = self.decoder_size
        input_size_LSTM_dec_intention_no_manoeuvre = self.soc_embedding_size + self.dyn_embedding_size

        if self.IA_module:
            # Decoder LSTM
            self.dec_lstm = torch.nn.LSTM(input_size_LSTM_dec_intention, hidden_size_LSTM_dec_intention)
        else:
            self.dec_lstm = torch.nn.LSTM(input_size_LSTM_dec_intention_no_manoeuvre, hidden_size_LSTM_dec_intention)

        input_size_LSTM_dec = self.targ_enc_size + self.fuse_enc_size + self.num_lat_classes + self.num_lon_classes
        hidden_size_LSTM_dec = self.decoder_size

        # Bidirectional LSTM to predict the future ...
        self.dec_lstm = nn.LSTM(input_size=input_size_LSTM_dec, hidden_size=hidden_size_LSTM_dec, bidirectional=True)  # biLSTM decoder

        # Add maneuvers 
        ## Add Linear layer to decode maneuvers.
        self.op = nn.Linear(self.decoder_size, self.output_dim)
        ''' Output layers '''
        ################################################
        # Start encoding the targetHistory.
        ################################################
    ## Start forward pass ...
    def forward(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask, lat_enc, lon_enc):
        ''' Forward target vehicle's dynamic'''

        # 1. TemporalConvolution : 1D-Conv to capture the temporal consistency.
        #    Return ->    dyn_enc :  [405, 32, 16] / tempEmb == 32 
        # 2. Activate LeakyReLU(dyn_enc) : 
        #    Return ->    dyn_enc :  [405, 32, 16]
        # 3. GRU [32, 64] :   nn.GRU(input_size=32, hidden_size=64, num_layers=1)
        # 4. Output, dyn_enc = GRU(dyn_enc.permute(1, 2, 0))
        #    Return ->    output [16, 405, 64] / and dyn_enc [1, 405, 64] 
	
	# Insert the temporal consistency in the dynamic encoding info.
        dyn_enc = self.leaky_relu(self.temporalConv(targsHist.permute(1,2,0))) #   torch.Size([n_samples, 32, 16]) 
                                                                               # /e.g., torch.Size([405, 32, 16])
	
        dyn_enc, (h_n_, c_n_) = self.nbh_bilstm(dyn_enc)  # BiLSTM
        u_dyn_enc = dyn_enc[:, :, :self.encoder_size] + dyn_enc[:, :, self.encoder_size:]
        dyn_enc = u_dyn_enc.permute(2,0,1)
        # The hidden states are then activated.
        dyn_enc = self.leaky_relu(self.dyn_emb(dyn_enc.view(dyn_enc.shape[1],dyn_enc.shape[2])))
        ''' Forward neighbour vehicles'''
        nbrs_enc = self.leaky_relu(self.temporalConv(nbsHist.permute(1, 2, 0))) 
        ## Output and nbrs_enc are the encoded neighbors' vehicles...
	# Once activated, we then can set the neighbors' encoded info into GRU layer.
        output, nbrs_enc = self.nbh_GRU(nbrs_enc.permute(2, 0, 1)) 
	#h_dec, (h_n, c_n) = self.dec_lstm(enc)
        #u_emb_batch = h_dec[:, :, :self.decoder_size] + h_dec[:, :, self.decoder_size:]
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #_, (nbrs_enc, _) = self.nbh_lstm(nbrs_enc.permute(2, 0, 1))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])  # Resize it baby
        ##################################################
        # Let's mask the shiiiiiiiiiiiiiiiiiit! 
        ##################################################
        ''' Masked neighbour vehicles'''
        # 1. Init the mask and set float() or float32() to fit PyTorch data form.
        nbrs_grid = torch.zeros_like(nbsMask).float()
        nbrs_grid = nbrs_grid.masked_scatter_(nbsMask, nbrs_enc)
        nbrs_grid = nbrs_grid.permute(0,3,2,1)
        nbrs_grid = self.nbrs_conv_social(nbrs_grid)
        ###################################################
        # End encoding the neighboringHistory.
        ###################################################
        merge_grid = self.pool_after_merge(nbrs_grid)  
        # . Reshape/ adjust inputs ...
        social_context = merge_grid.view(-1, self.social_context_size)
        ## Concatenate the social context ... surrounding vehicle and ego-vehicle ...
        '''Concatenate social_context (neighbors + ego's planning) and dyn_enc, then place into the targsEncMask '''
    
        target_enc = torch.cat((social_context, dyn_enc),1)   
        # <------------------------------>#<---------------------------<----->|
        #    n_Hist                       .   n_Fut                    |      | Past/Fut
        # +-------------------------------+---------------------------><------|
        # |nbsHist|nbsHist| [16, nb_nbs, 2].                           |      |  
        # +-------------------------------+.                           |inputs| Tensors
        #|targsHist|       [16, nb_targs,2]. |nbsFut|[25, nb_targs, 2 ]|      | 
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
        target_grid = torch.zeros_like(targsEncMask).float()
        target_grid = target_grid.masked_scatter_(targsEncMask, target_enc)
        if self.use_fusion:
            '''Fully Convolutional network to get a grid to be fused'''
            fuse_conv1 = self.relu(self.fcn_conv1(target_grid.permute(0, 3, 2, 1)))
            fuse_conv1 = self.bn1(fuse_conv1)
            fuse_conv1 = self.dropout1(fuse_conv1)
            #fuse_conv1 = self.fcn_pool1(fuse_conv1) # Don't pool baby!
            
            fuse_conv2 = self.leaky_relu(self.fcn_conv2(fuse_conv1))  # LeakyRelu activation function has been added here 
            fuse_conv2 = self.bn2(fuse_conv2)
            fuse_conv2 = self.dropout2(fuse_conv2)
            #fuse_conv2 = self.fcn_pool2(fuse_conv2)
            
            fuse_trans1 = self.leaky_relu(self.fcn_convTrans1(fuse_conv2))
            fuse_trans1 = self.back_bn1(fuse_trans1+fuse_conv1)
            fuse_trans1 = self.dropout3(fuse_trans1)
            
            fuse_trans2 = self.leaky_relu(self.fcn_convTrans2(fuse_trans1))
            fuse_trans2 = self.back_bn2(fuse_trans2)
            fuse_trans2 = self.dropout4(fuse_trans2)
            # Extract the location with targets
            fuse_grid_mask = targsEncMask[:,:,:,0:self.fuse_enc_size]
            fuse_grid = torch.zeros_like(fuse_grid_mask).float()
            fuse_grid = fuse_grid.masked_scatter_(fuse_grid_mask, fuse_trans2.permute
