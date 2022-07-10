import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_dim).requires_grad_()
        out, (_) = self.rnn(X, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

    def __str__(self) -> str:
        return "RNN"
