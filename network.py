import torch.nn as nn

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.nestfunc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(12), 
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=6, stride=6),
            
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=24*4*4, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
        )
        
    def forward(self, t):
        t = self.nestfunc(t)
        return t