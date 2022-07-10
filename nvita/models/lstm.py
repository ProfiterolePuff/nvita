import torch
import torch.nn as nn

from blitz.modules import BayesianLSTM

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = BayesianLSTM(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_dim).requires_grad_()
        out, (_, _) = self.lstm(X, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def __str__(self) -> str:
        return "LSTM"