import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonGNN(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=16, num_nodes=553):
        super().__init__()

        self.num_nodes  = num_nodes
        self.hidden_dim = hidden_dim

        # learnable adjacency
        A_init = torch.eye(num_nodes) + 0.01 * torch.rand(num_nodes, num_nodes)
        self.A = nn.Parameter(A_init)

        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, hidden_dim)
        self.residual = nn.Linear(input_dim, hidden_dim)

        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.residual]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)

    def forward(self, x):

        # x: (batch, frames, nodes, 3)
        b, f, n, c = x.shape

        res = self.residual(x)                          # (b, f, n, hidden)

        x = F.relu(self.norm1(self.fc1(x)))             # (b, f, n, hidden)

        A = torch.softmax(self.A, dim=-1)               # (n, n)
        x = torch.einsum("ij, bfjd -> bfid", A, x)     # (b, f, n, hidden)

        x = F.relu(self.norm2(self.fc2(x) + res))      # (b, f, n, hidden)

        x = self.dropout(x)

        # normalize before flatten to prevent magnitude explosion
        x = x / (self.num_nodes ** 0.5)

        # (batch, frames, nodes * hidden)
        x = x.reshape(b, f, n * self.hidden_dim)

        return x