import torch  
from torch import nn  
import torch.nn.functional as F  
import math  
import numpy as np  
import matplotlib.pyplot as plt  

def scaled_dot_product(q, k, v, mask=None):
    """
    Calculate the scaled dot-product attention.
    """
    d_k = q.size()[-1]  # Get the dimension of the key vector
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # Compute attention logits by matrix multiplication and scaling

    if mask is not None:  # If a mask is provided
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))  # Apply the mask, setting certain logits to negative infinity

    attention = F.softmax(attn_logits, dim=-1)  # Apply softmax to get attention weights
    values = torch.matmul(attention, v)  # Compute the weighted sum of values

    return values, attention  # Return the attention-applied values and the attention map

class MultiheadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()  # Call the parent class's constructor
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."  # Ensure embedding dimension is divisible by the number of heads
        
        self.embed_dim = embed_dim  # Store the embedding dimension
        self.num_heads = num_heads  # Store the number of attention heads
        self.head_dim = embed_dim // num_heads  # Compute the dimension of each head
        
        # Linear layers for Q, K, V projections
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)  # Define a linear layer to project input to queries, keys, and values
        self.o_proj = nn.Linear(embed_dim, embed_dim)  # Define a linear layer for output projection
        
        self._reset_parameters()  # Initialize the parameters

    def _reset_parameters(self):
        """
        Initialize the parameters.
        """
        nn.init.xavier_uniform_(self.qkv_proj.weight)  # Initialize QKV projection weights with Xavier uniform distribution
        self.qkv_proj.bias.data.fill_(0)  # Initialize the bias for QKV projection to zero
        nn.init.xavier_uniform_(self.o_proj.weight)  # Initialize output projection weights with Xavier uniform distribution
        self.o_proj.bias.data.fill_(0)  # Initialize the bias for output projection to zero
   
    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass for multi-head attention.
        """
        batch_size, seq_length, embed_dim = x.size()  # Get the dimensions of the input tensor
        
        # Linear projection and reshape for Q, K, V
        qkv = self.qkv_proj(x).reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)  # Project and reshape input to Q, K, V tensors
        qkv = qkv.permute(0, 2, 1, 3)  # Rearrange the dimensions to [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)  # Split the QKV tensor into queries, keys, and values
        
        # Scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask=mask)  # Compute the scaled dot-product attention
        
        # Concatenate heads and project
        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)  # Reorder and reshape the attention-applied values
        output = self.o_proj(values)  # Apply the output projection
        
        if return_attention:  # If attention maps are to be returned
            return output, attention  # Return the output and attention maps
        else:
            return output  # Return only the output

class EncoderBlock(nn.Module):
    """
    Single encoder block with self-attention and feedforward network.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.5):
        super().__init__()  # Call the parent class's constructor
        
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)  # Initialize the self-attention mechanism
        
        # Feedforward network
        self.feedforward = nn.Sequential(  # Define a sequence of layers for the feedforward network
            nn.Linear(input_dim, dim_feedforward),  # Linear layer to expand dimensions
            nn.Dropout(dropout),  # Apply dropout for regularization
            nn.ReLU(inplace=True),  # Apply ReLU activation in place
            nn.Linear(dim_feedforward, input_dim)  # Linear layer to project back to input dimensions
        )

        self.norm1 = nn.LayerNorm(input_dim)  # Layer normalization after self-attention
        self.norm2 = nn.LayerNorm(input_dim)  # Layer normalization after feedforward network
        self.dropout = nn.Dropout(dropout)  # Dropout layer for additional regularization

    def forward(self, x, mask=None):
        """
        Forward pass for the encoder block.
        """
        # Self-attention with residual connection and normalization
        attn_out = self.self_attn(x, mask=mask)  # Compute the self-attention output
        x = self.norm1(x + self.dropout(attn_out))  # Apply residual connection, dropout, and normalization
        
        # Feedforward with residual connection and normalization
        ff_out = self.feedforward(x)  # Compute the feedforward network output
        x = self.norm2(x + self.dropout(ff_out))  # Apply residual connection, dropout, and normalization

        return x  # Return the output of the encoder block

class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of multiple encoder blocks.
    """
    def __init__(self, num_layers, **block_args):
        super().__init__()  # Call the parent class's constructor
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])  # Create a list of encoder blocks

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder.
        """
        for layer in self.layers:  # Iterate over each encoder block
            x = layer(x, mask=mask)  # Pass the input through the current encoder block
        return x  # Return the final output of the encoder

    def get_attention_maps(self, x, mask=None):
        """
        Retrieve attention maps from each layer.
        """
        attention_maps = []  # Initialize a list to store attention maps
        for layer in self.layers:  # Iterate over each encoder block
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)  # Get the attention map from the current block
            attention_maps.append(attn_map)  # Add the attention map to the list
            x = layer(x)  # Pass the input through the current encoder block
        return attention_maps  # Return the list of attention maps

class ProjectorBlock(nn.Module):
    """
    Projection block for spatial transformations.
    """
    def __init__(self, in_features, out_features):
        super().__init__()  # Call the parent class's constructor
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)  # Define a 2D convolutional layer
        
    def forward(self, x):
        return self.op(x)  # Apply the convolutional layer to the input

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism.
    """
    def __init__(self, in_features, normalize_attn=True):
        super().__init__()  # Call the parent class's constructor
        self.normalize_attn = normalize_attn  # Store whether to normalize attention
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)  # Define a 2D convolutional layer for attention
        
    def forward(self, l, g):
        """
        Forward pass for spatial attention.
        """
        N, C, H, W = l.size()  # Get the dimensions of the input tensor
        c = self.op(l + g)  # Apply the convolutional operation to the sum of inputs
        
        if self.normalize_attn:  # If attention should be normalized
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, H, W)  # Apply softmax to normalize attention
        else:
            a = torch.sigmoid(c)  # Apply sigmoid for attention

        g = torch.mul(a.expand_as(l), l)  # Compute the weighted sum of features
        if self.normalize_attn:  # If attention is normalized
            g = g.view(N, C, -1).sum(dim=2)  # Sum across spatial dimensions
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)  # Apply adaptive average pooling

        return c.view(N, 1, H, W), g  # Return the attention map and the output

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism.
    """
    def __init__(self, hidden_size):
        super().__init__()  # Call the parent class's constructor
        self.hidden_size = hidden_size  # Store the hidden size
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # Define a linear layer for score computation
        self.fc2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)  # Define a linear layer for attention computation

    def forward(self, hidden_states):
        """
        Forward pass for temporal attention.
        """
        score_first_part = self.fc1(hidden_states)  # Compute the first part of the attention score
        h_t = hidden_states[:, -1, :]  # Get the hidden state of the last time step
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)  # Compute the attention score
        attention_weights = F.softmax(score, dim=1)  # Apply softmax to get attention weights
        context_vector = torch.bmm(hidden_states.permute(0, 2, 1), attention_weights.unsqueeze(2)).squeeze(2)  # Compute the context vector
        pre_activation = torch.cat((context_vector, h_t), dim=1)  # Concatenate the context vector and the last hidden state
        attention_vector = torch.tanh(self.fc2(pre_activation))  # Apply the second linear layer and tanh activation

        return attention_vector, attention_weights  # Return the attention vector and attention weights

def plot_attention_maps(input_data, attn_maps, idx=0):
    """
    Plot attention maps to visualise the attention mechanism.
    """
    if input_data is not None:  # If input data is provided
        input_data = input_data[idx].detach().cpu().numpy()  # Convert the input data to a NumPy array
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])  # Create a range of sequence indices
    
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]  # Convert each attention map to a NumPy array
    num_heads = attn_maps[0].shape[0]  # Get the number of attention heads
    num_layers = len(attn_maps)  # Get the number of layers
    seq_len = input_data.shape[0]  # Get the sequence length
    fig_size = 4 if num_heads == 1 else 3  # Set the figure size based on the number of heads
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size))  # Create subplots
    
    if num_layers == 1:  # If there is only one layer
        ax = [ax]  # Wrap the axis in a list
    if num_heads == 1:  # If there is only one head
        ax = [[a] for a in ax]  # Wrap the axis in a nested list
    
    for row in range(num_layers):  # Iterate over each layer
        for column in range(num_heads):  # Iterate over each attention head
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)  # Display the attention map
            ax[row][column].set_xticks(list(range(seq_len)))  # Set the x-axis ticks
            ax[row][column].set_xticklabels(input_data.tolist())  # Set the x-axis tick labels
            ax[row][column].set_yticks(list(range(seq_len)))  # Set the y-axis ticks
            ax[row][column].set_yticklabels(input_data.tolist())  # Set the y-axis tick labels
            ax[row][column].set_title(f"Layer {row + 1}, Head {column + 1}")  # Set the title for each subplot
    
    fig.subplots_adjust(hspace=0.5)  # Adjust the spacing between subplots
    plt.show()  # Display the plot
