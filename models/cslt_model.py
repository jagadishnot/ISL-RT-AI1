import torch
import torch.nn as nn
import math
from models.gnn import SkeletonGNN


# ---------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------
# Temporal Convolution Block
# ---------------------------------------------------------

class TemporalBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.conv1   = nn.Conv1d(dim, dim, 3, padding=1)
        self.conv2   = nn.Conv1d(dim, dim, 5, padding=2)
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        self.relu    = nn.ReLU()

    def forward(self, x):

        residual = x
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x) + self.conv2(x))
        x = x.transpose(1, 2)
        x = self.norm(x + residual)
        x = self.dropout(x)

        return x


# ---------------------------------------------------------
# CSLT Model with GNN
# ---------------------------------------------------------

class CSLTModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # 478 face + 21 left + 21 right + 33 pose = 553
        self.num_nodes  = 553
        self.node_in    = 3
        self.hidden_dim = 16      # 553 * 16 = 8848 → manageable
        model_dim       = 256

        # -------------------------------------------------
        # GNN
        # -------------------------------------------------

        self.gnn = SkeletonGNN(
            input_dim=self.node_in,
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes
        )

        # -------------------------------------------------
        # Projection: (553 * 16) → 256
        # -------------------------------------------------

        self.embedding = nn.Sequential(
            nn.Linear(self.num_nodes * self.hidden_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # -------------------------------------------------
        # Positional Encoding
        # -------------------------------------------------

        self.pos = PositionalEncoding(model_dim)

        # -------------------------------------------------
        # Temporal Blocks
        # -------------------------------------------------

        self.temporal1 = TemporalBlock(model_dim)
        self.temporal2 = TemporalBlock(model_dim)

        # -------------------------------------------------
        # Transformer
        # -------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # -------------------------------------------------
        # Classifier
        # -------------------------------------------------

        self.classifier = nn.Linear(model_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):

        # small gain for large input projection
        nn.init.xavier_uniform_(self.embedding[0].weight, gain=0.1)
        nn.init.zeros_(self.embedding[0].bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):

        # x: (batch, frames, 1659)
        b, f, _ = x.shape

        # reshape for GNN → (batch, frames, 553, 3)
        x = x.view(b, f, self.num_nodes, self.node_in)

        # GNN → (batch, frames, 553 * 16)
        x = self.gnn(x)

        # projection → (batch, frames, 256)
        x = self.embedding(x)

        x = self.pos(x)

        x = self.temporal1(x)
        x = self.temporal2(x)

        x = self.transformer(x)

        x = self.classifier(x)

        return x