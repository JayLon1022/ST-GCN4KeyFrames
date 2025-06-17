import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml

class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj):
        # x: [B, N, C], adj: [B, N, N]
        h = torch.bmm(adj, x)  # [B, N, C]
        h = self.gcn(h)
        h = self.bn(h.transpose(1,2)).transpose(1,2)
        h = self.relu(h)
        res = self.residual(x)
        return h + res

class STGCN(pl.LightningModule):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        in_channels = cfg['in_channels']
        hidden_channels = cfg['hidden_channels']
        num_layers = cfg['num_layers']
        self.save_hyperparameters()

        layers = []
        last_c = in_channels
        for c in hidden_channels:
            layers.append(GraphConvBlock(last_c, c))
            last_c = c
        self.gcn_layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(last_c, last_c)

    def forward(self, x, adj):
        # x: [B, N, C], adj: [B, N, N]
        for gcn in self.gcn_layers:
            x = gcn(x, adj)
        out = self.out_proj(x)
        return out  # [B, N, C] 