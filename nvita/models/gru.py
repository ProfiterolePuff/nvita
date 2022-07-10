import torch.nn as nn

from blitz.modules import BayesianGRU

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = BayesianGRU(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        out, _ = self.gru(X)
        out = self.fc(out[:, -1, :]) 
        return out
    
    def __str__(self) -> str:
        return "GRU"