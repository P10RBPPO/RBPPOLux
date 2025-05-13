import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        
        # Modular network size
        network_dim = 75
        
        self.layer1 = nn.Linear(in_dim, network_dim)
        self.layer2 = nn.Linear(network_dim, network_dim)
        self.layer3 = nn.Linear(network_dim, out_dim)
        
    def forward(self, obs):
        # Convert to tensor, if numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        
        return output