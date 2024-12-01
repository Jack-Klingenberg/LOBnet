import torch
from torch import nn

# Components for DeepLOB Model (https://arxiv.org/pdf/1808.03668) 
# Parameters allow some customization while keeping the structure of the DeepLOB paper's model intact
class DeepLOB_ConvolutionalBlock(nn.Module):
    def __init__(self, input_depth=1, output_depth=16, activation=nn.LeakyReLU, slope=.01):
        super().__init__()
        self.slope = slope # slope for LeakyReLU

        self.conv1 = nn.Conv2d(input_depth, output_depth, kernel_size=(1,2), stride=(1,2))
        self.conv2 = nn.Conv2d(output_depth, output_depth, kernel_size=(4,1))
        self.conv3 = nn.Conv2d(output_depth, output_depth, kernel_size=(4,1)) 

        self.bn1 = nn.BatchNorm2d(output_depth)
        self.bn2 = nn.BatchNorm2d(output_depth)
        self.bn3 = nn.BatchNorm2d(output_depth)
        
        self.act1,self.act2,self.act3 = None,None,None
        if activation == nn.LeakyReLU:
            self.act1 = nn.LeakyReLU(negative_slope=self.slope)
            self.act2 = nn.LeakyReLU(negative_slope=self.slope)
            self.act3 = nn.LeakyReLU(negative_slope=self.slope)
        elif activation == nn.Tanh:
            self.act1 = nn.Tanh()
            self.act2 = nn.Tanh()
            self.act3 = nn.Tanh()
        else:
            raise ValueError("Unknown activation not nn.LeakyReLU or nn.Tanh")

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)
        y = self.bn1(y)

        y = self.conv2(y)
        y = self.act2(y)
        y = self.bn2(y)

        y = self.conv3(y)
        y = self.act3(y)
        y = self.bn3(y)

        return y
    
class DeepLOB_InceptionConvSubunit(nn.Module):
    def __init__(self, input_depth=16, output_depth=32, slope=.01, kernel_size = (3,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_depth, out_channels=output_depth, kernel_size=(1,1), padding="same")
        self.act1 = nn.LeakyReLU(slope)
        self.bn1 = nn.BatchNorm2d(output_depth)

        self.conv2 = nn.Conv2d(in_channels=output_depth, out_channels=output_depth, kernel_size=kernel_size, padding='same')
        self.act2 = nn.LeakyReLU(slope)
        self.bn2 = nn.BatchNorm2d(output_depth)
    def forward(self,x):
        y = self.bn1(self.act1(self.conv1(x)))
        y = self.bn2(self.act2(self.conv2(y)))

        return y
    
class DeepLOB_IncpetionPoolSubunut(nn.Module):
    def __init__(self, input_depth=16, output_depth=32, slope=.01):
        super().__init__()
        self.pool1 = nn.MaxPool2d((3,1), stride=(1,1), padding=(1,0))
        self.conv1 = nn.Conv2d(in_channels=input_depth, out_channels=output_depth, kernel_size=(1,1), padding='same')
        self.act1 = nn.LeakyReLU(slope)
        self.bn1 = nn.BatchNorm2d(output_depth)

    def forward(self,x):
        y = self.bn1(self.act1(self.conv1(self.pool1(x))))
        
        return y
            
class DeepLOB_Network_v1(nn.Module):
    def __init__(self, y_len, device):
        super().__init__()
        self.device = device
        self.conv_block_1 = DeepLOB_ConvolutionalBlock(input_depth=1, output_depth=16)
        self.conv_block_2 = DeepLOB_ConvolutionalBlock(input_depth=16, output_depth=16)
        self.conv_block_3 = DeepLOB_ConvolutionalBlock(input_depth=16, output_depth=16)

        self.incep1 = DeepLOB_InceptionConvSubunit(kernel_size=(3,1))
        self.incep2 = DeepLOB_InceptionConvSubunit(kernel_size=(5,1))
        self.incep3 = DeepLOB_IncpetionPoolSubunut()

        self.lstm1 = nn.LSTM(input_size=96*5, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, y_len)

    def forward(self, x):
        y = self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))
        y = torch.cat((self.incep1(y), self.incep2(y), self.incep3(y)), dim=1)
        y = y.permute(0, 2, 1, 3)  # [32, 82, 96, 5]
        
        # reshape to (batch_size, sequence_length, features)
        batch_size = y.shape[0]
        y = y.reshape(batch_size, y.shape[1], -1)  # combine last two dimensions
        
        h0 = torch.zeros(1, batch_size, 64).to(self.device)
        c0 = torch.zeros(1, batch_size, 64).to(self.device)
        
        y, _ = self.lstm1(y, (h0, c0))
        y = self.fc1(y[:, -1, :])
        return y


# Multiheaded attention before LSTM
class DeepLOB_Network_v2(nn.Module):
    def __init__(self, y_len, device):
        super().__init__()
        self.device = device
        self.conv_block_1 = DeepLOB_ConvolutionalBlock(input_depth=1, output_depth=16)
        self.conv_block_2 = DeepLOB_ConvolutionalBlock(input_depth=16, output_depth=16)
        self.conv_block_3 = DeepLOB_ConvolutionalBlock(input_depth=16, output_depth=16)

        self.incep1 = DeepLOB_InceptionConvSubunit(kernel_size=(3,1))
        self.incep2 = DeepLOB_InceptionConvSubunit(kernel_size=(5,1))
        self.incep3 = DeepLOB_IncpetionPoolSubunut()

        self.attention = nn.MultiheadAttention(embed_dim=96*5, num_heads=8, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=96*5, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, y_len)

    def forward(self, x):
        y = self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))
        y = torch.cat((self.incep1(y), self.incep2(y), self.incep3(y)), dim=1)
        y = y.permute(0, 2, 1, 3)  # [32, 82, 96, 5]
        
        # reshape to (batch_size, sequence_length, features)
        batch_size = y.shape[0]
        y = y.reshape(batch_size, y.shape[1], -1)  # combine last two dimensions
        
        y, _ = self.attention(y, y, y)

        h0 = torch.zeros(1, batch_size, 64).to(self.device)
        c0 = torch.zeros(1, batch_size, 64).to(self.device)
        
        y, _ = self.lstm1(y, (h0, c0))
        y = self.fc1(y[:, -1, :])
        return y
