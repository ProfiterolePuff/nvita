import torch.nn as nn

from blitz.modules import BayesianConv1d

class CNN(nn.Module):
    def __init__(self, training_length, c, f0, f1, f2):
        super(CNN,self).__init__()
        self.conv1d = BayesianConv1d(training_length, c, kernel_size = 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(f0, 1)
        self.fc1 = nn.Linear(f1, f2)
        self.fc2 = nn.Linear(f2, 1)
        
    def forward(self, X):
        out = self.conv1d(X)
        out = self.relu(out)
        out = self.fc0(out)
        out = out[:, :, -1]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

    def __str__(self) -> str:
        return "CNN"