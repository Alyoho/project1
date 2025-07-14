import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader



class LSTMModel_S(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, input_seq_len, output_seq_len):
        super(LSTMModel_S, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Linear Layer to generate the output for each time step
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply linear layer to each time step in the sequence
        out = self.linear(out)  # Shape: [batch_size, seq_len, output_dim]
        
        return out

class LSTMModel_L(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, input_seq_len, output_seq_len):
        super(LSTMModel_L, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.output_dim = output_dim
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Linear Layer to generate the output for each time step
        self.linear = nn.Linear(hidden_dim, output_dim)

        # Fully connected layer to map from input_seq_len to output_seq_len
        self.fc = nn.Linear(input_seq_len, output_seq_len)
    
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply linear layer to each time step in the sequence
        out = self.linear(out)  # Shape: [batch_size, input_seq_len, output_dim]
        
        # Map output sequence length from input_seq_len (96) to output_seq_len (240)
        out = out.contiguous().view(out.size(0), -1)  # Flatten to [batch_size, input_seq_len * output_dim]
        out = self.fc(out)  # Shape: [batch_size, output_seq_len]
        
        # Reshape output to [batch_size, output_seq_len, output_dim]
        out = out.view(out.size(0), self.output_seq_len, self.output_dim)
        
        return out  # Output shape: [batch_size, output_seq_len, output_dim]