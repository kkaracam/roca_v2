import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import fps

class STN3D(nn.Module):
    def __init__(self, input_channels=3):
        super(STN3D, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(self.input_channels).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x

class TargetEncoder(nn.Module):
    def __init__(self, embedding_size, input_channels=3):
        super(TargetEncoder, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        x_ftrs = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc(x_ftrs)
        return x#, x_ftrs
    def latent_from_feats(self, x_ftrs):
        return self.fc(x_ftrs)

class TargetEncoder2(nn.Module):
    def __init__(self, embedding_size, input_channels=3):
        super(TargetEncoder2, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024*3, embedding_size)

    def forward(self, x1):
        x2 = torch.stack([_x[fps(_x, ratio=0.25)] for _x in x1])
        x3 = torch.stack([_x[fps(_x, ratio=0.25)] for _x in x2])
        
        out = []
        for x in (x1,x2,x3):
            batch_size = x.shape[0]
            num_points = x.shape[1]

            x = x[:, :, : self.input_channels]
            x = x.transpose(2, 1)  # transpose to apply 1D convolution
            x = self.mlp1(x)
            x = self.mlp2(x)

            x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
            out.append(x)
        x = torch.cat(out,dim=-1)
        x = self.fc(x)
        return x

class TargetDecoder(nn.Module):
    def __init__(self, input_channels=3, num_points = 1024):
        super(TargetDecoder, self).__init__()
        self.input_channels = input_channels
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points*3),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp(x).view(batch_size, self.num_points, 3)
        return x-0.5

class ParamDecoder2(nn.Module):
    def __init__(self, input_dim, intermediate_layer, embedding_size, use_bn=False):
        super(ParamDecoder2, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, intermediate_layer)
        self.fc3 = nn.Linear(intermediate_layer, embedding_size)
        if (use_bn):
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(intermediate_layer)

    def forward(self, x, use_bn=False):
        if use_bn:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = self.fc1(x)
            x = self.fc2(x)

        x = self.fc3(x)

        return x