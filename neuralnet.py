import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=46592, out_features=hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_size, out_features=400),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=400, out_features=200),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=200, out_features=n_actions),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
