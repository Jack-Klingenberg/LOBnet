import torch
from torch import nn
import torch.nn.functional as F

"""
CNN based feature extraction with a transformer encoder aimed at capturing 
local patterns and long-range dependencies in LOB data.

Architecture decisions:
- Uses CNN for initial dimensionality reduction and feature extraction
- For transformer, small d_model (32) and heads (2) to prevent overfitting (also dropout)
""" 
class TransformerLOB(nn.Module):
    def __init__(self, y_len=3, device='cpu'):
        super().__init__()
        
        self.input_dim = 40  
        self.d_model = 32      
        self.nhead = 2 
        self.num_layers = 3 # number of transformer layers
        self.dropout = 0.1 # slight dropout to help prevent overfitting
        
        # CNN reduces sequence length by half while learning useful features
        self.conv_reduction = nn.Sequential(
            nn.Conv1d(self.input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # reduce sequence length by 2
            PermuteLayer((0, 2, 1)), # (batch, seq_len, 32)
            nn.LayerNorm(32), # normalize across channels (C = 32)
            PermuteLayer((0, 2, 1)) # (batch, 32, seq_len)
        )
        
        self.input_projection = nn.Linear(32, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
    
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead, 
            dropout=self.dropout,
            batch_first=True,
            dim_feedforward=64,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        self.fc1 = nn.Linear(self.d_model, 64)  
        self.fc2 = nn.Linear(64, y_len) 
        
    def forward(self, x):
        x = x.squeeze(1) 
        
        x = x.transpose(1, 2) # (batch, channels, seq_len)
        x = self.conv_reduction(x)
        x = x.transpose(1, 2) # (batch, seq_len, features)
        
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class PermuteLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims) 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

"""
Baseline DeepLOB Network. More direct implementation compared to first implementation. Does not allow for architecture changes.
Shares the CNN --> inception --> LSTM architecture. 
"""
class BaselineDeepLOB(nn.Module):
    def __init__(self, y_len,device):
        super().__init__()
        self.y_len = y_len
        self.device = device
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        
        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(self.device)
        c0 = torch.zeros(1, x.size(0), 64).to(self.device)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        
#         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        #forecast_y = torch.softmax(x, dim=1)
        
        return x

class DeepLOB_Network_v2(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        
        # Convolutional Feature Extractors
        self.conv1_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2))
        self.conv1_activation1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv1_bn1 = nn.BatchNorm2d(32)
        self.conv1_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))
        self.conv1_activation2 = nn.LeakyReLU(negative_slope=0.01)
        self.conv1_bn2 = nn.BatchNorm2d(32)
        self.conv1_layer3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))
        self.conv1_activation3 = nn.LeakyReLU(negative_slope=0.01)
        self.conv1_bn3 = nn.BatchNorm2d(32)

        self.conv2_layer1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2))
        self.conv2_activation1 = nn.Tanh()
        self.conv2_bn1 = nn.BatchNorm2d(32)
        self.conv2_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))
        self.conv2_activation2 = nn.Tanh()
        self.conv2_bn2 = nn.BatchNorm2d(32)
        self.conv2_layer3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))
        self.conv2_activation3 = nn.Tanh()
        self.conv2_bn3 = nn.BatchNorm2d(32)

        self.conv3_layer1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10))
        self.conv3_activation1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv3_bn1 = nn.BatchNorm2d(32)
        self.conv3_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))
        self.conv3_activation2 = nn.LeakyReLU(negative_slope=0.01)
        self.conv3_bn2 = nn.BatchNorm2d(32)
        self.conv3_layer3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1))
        self.conv3_activation3 = nn.LeakyReLU(negative_slope=0.01)
        self.conv3_bn3 = nn.BatchNorm2d(32)

        # Inception Block
        self.inp1_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same')
        self.inp1_activation1 = nn.LeakyReLU(negative_slope=0.01)
        self.inp1_bn1 = nn.BatchNorm2d(64)
        self.inp1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same')
        self.inp1_activation2 = nn.LeakyReLU(negative_slope=0.01)
        self.inp1_bn2 = nn.BatchNorm2d(64)

        self.inp2_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same')
        self.inp2_activation1 = nn.LeakyReLU(negative_slope=0.01)
        self.inp2_bn1 = nn.BatchNorm2d(64)
        self.inp2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same')
        self.inp2_activation2 = nn.LeakyReLU(negative_slope=0.01)
        self.inp2_bn2 = nn.BatchNorm2d(64)

        self.inp3_pool = nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0))
        self.inp3_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same')
        self.inp3_activation = nn.LeakyReLU(negative_slope=0.01)
        self.inp3_bn = nn.BatchNorm2d(64)

        self.dropout1 = nn.Dropout(dropout_rate)  # After first conv block
        self.dropout2 = nn.Dropout(dropout_rate)  # After second conv block
        self.dropout3 = nn.Dropout(dropout_rate)  # After third conv block
        self.dropout4 = nn.Dropout(dropout_rate)  # After inception module
        self.dropout_fc = nn.Dropout(dropout_rate)  # Between FC layers

        # Single directional LSTM
        self.lstm = nn.LSTM(
            input_size=192,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),  # 128 from bidirectional LSTM
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Output layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

        # LSTM and Fully Connected Layers
        #self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        #self.fc1 = nn.Linear(64, self.num_classes)
    
    def forward(self, input_tensor):
        hidden_state = torch.zeros(1, input_tensor.size(0), 64).to(input_tensor.device)
        cell_state = torch.zeros(1, input_tensor.size(0), 64).to(input_tensor.device)

        # Feature Extraction 
        x = self.conv1_layer1(input_tensor)
        x = self.conv1_activation1(self.conv1_bn1(x))
        x = self.conv1_layer2(x)
        x = self.conv1_activation2(self.conv1_bn2(x))
        x = self.conv1_layer3(x)
        x = self.conv1_activation3(self.conv1_bn3(x))
        x = self.dropout1(x)

        x = self.conv2_layer1(x)
        x = self.conv2_activation1(self.conv2_bn1(x))
        x = self.conv2_layer2(x)
        x = self.conv2_activation2(self.conv2_bn2(x))
        x = self.conv2_layer3(x)
        x = self.conv2_activation3(self.conv2_bn3(x))
        x = self.dropout2(x)

        x = self.conv3_layer1(x)
        x = self.conv3_activation1(self.conv3_bn1(x))
        x = self.conv3_layer2(x)
        x = self.conv3_activation2(self.conv3_bn2(x))
        x = self.conv3_layer3(x)
        x = self.conv3_activation3(self.conv3_bn3(x))
        x = self.dropout3(x)

        # Inception Blocks
        x_incep1 = self.inp1_conv1(x)
        x_incep1 = self.inp1_activation1(self.inp1_bn1(x_incep1))
        x_incep1 = self.inp1_conv2(x_incep1)
        x_incep1 = self.inp1_activation2(self.inp1_bn2(x_incep1))

        x_incep2 = self.inp2_conv1(x)
        x_incep2 = self.inp2_activation1(self.inp2_bn1(x_incep2))
        x_incep2 = self.inp2_conv2(x_incep2)
        x_incep2 = self.inp2_activation2(self.inp2_bn2(x_incep2))

        x_incep3 = self.inp3_conv(x)
        x_incep3 = self.inp3_activation(self.inp3_bn(x_incep3))
        x_incep3 = self.inp3_pool(x_incep3)

        x = torch.cat((x_incep1, x_incep2, x_incep3), dim=1)
        x = self.dropout4(x) 



        # # LSTM and fully connected layers
        # x, _ = self.lstm(x, (hidden_state, cell_state))
        # x = self.fc1(x[:, -1, :])
        # forecast_output = torch.softmax(x, dim=1)

        # return forecast_output
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)
        batch_size, seq_len, channels, height = x.shape
        x = x.reshape(batch_size, seq_len, channels * height)
        
        # Bidirectional LSTM
        x, _ = self.lstm(x, (hidden_state, cell_state))
        
        # Final prediction
        x = self.fc1(x[:, -1, :]) 
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x
        
        #return torch.softmax(x, dim=1)