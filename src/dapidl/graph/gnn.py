"""Stage-2 clean learned-BANKSY: a small CNN on a tight nucleus crop (context-poor
nodes) + hand-rolled scatter-mean GraphSAGE (PyG is not a dependency)."""
from __future__ import annotations

import torch
from torch import nn


def scatter_mean(messages: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean of ``messages`` grouped by destination node ``index`` (here: aggregate
    each source node's neighbour messages). Nodes with no messages -> zeros."""
    d = messages.shape[1]
    out = torch.zeros(num_nodes, d, device=messages.device, dtype=messages.dtype)
    cnt = torch.zeros(num_nodes, 1, device=messages.device, dtype=messages.dtype)
    out.index_add_(0, index, messages)
    cnt.index_add_(0, index, torch.ones(index.shape[0], 1, device=messages.device, dtype=messages.dtype))
    return out / cnt.clamp_min(1.0)


class NucleusNodeCNN(nn.Module):
    """3 stride-2 conv blocks on a 1x40x40 nucleus crop -> GAP -> out_dim embedding."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.net = nn.Sequential(block(1, 32), block(32, 64), block(64, out_dim),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class SageLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, h, edge_index):
        src, dst = edge_index
        agg = scatter_mean(h[dst], src, num_nodes=h.shape[0])
        return self.lin(torch.cat([h, agg], dim=1))


class SageCellTyper(nn.Module):
    def __init__(self, node_dim=128, hidden=64, num_classes=4, layers=2):
        super().__init__()
        self.encoder = NucleusNodeCNN(out_dim=node_dim)
        dims = [node_dim] + [hidden] * layers
        self.sage = nn.ModuleList(SageLayer(dims[i], dims[i + 1]) for i in range(layers))
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, crops, edge_index):
        h = self.encoder(crops)
        for layer in self.sage:
            h = torch.relu(layer(h, edge_index))
        return self.head(h)


class NoGraphAggregator(nn.Module):
    """No-graph arm: ignores neighbours, returns zeros so the model sees only `se`."""
    needs_neighbours = False

    def forward(self, se, ne, valid):
        return torch.zeros_like(se)


class MeanAggregator(nn.Module):
    """Masked mean of neighbour features (the GraphSAGE-mean aggregation). `valid` is
    1.0 where a neighbour exists, 0.0 for -1 padding."""
    needs_neighbours = True

    def forward(self, se, ne, valid):
        cnt = valid.sum(1, keepdim=True).clamp_min(1.0)
        return (ne * valid[:, :, None]).sum(1) / cnt


class EdgeGATv2Aggregator(nn.Module):
    """GATv2 (Brody et al. 2022) attention over the k neighbour slots, with edge features
    in the score. Multi-head; heads*head_dim == node_dim so the output matches Mean/NoGraph
    width. needs_edge_attr -> GraphArmModel passes edge_attr (B,k,edge_dim)."""
    needs_neighbours = True
    needs_edge_attr = True

    def __init__(self, node_dim: int, edge_dim: int, heads: int = 4):
        super().__init__()
        assert node_dim % heads == 0, "node_dim must be divisible by heads"
        self.heads = heads
        self.hd = node_dim // heads
        self.w = nn.Linear(2 * node_dim + edge_dim, heads * self.hd)   # GATv2 shared transform
        self.a = nn.Parameter(torch.empty(heads, self.hd))             # per-head attention vector
        self.wv = nn.Linear(node_dim, heads * self.hd)                 # value projection
        nn.init.xavier_uniform_(self.a)

    def forward(self, se, ne, valid, edge_attr):
        b, k, _ = ne.shape
        se_exp = se[:, None, :].expand(-1, k, -1)                      # [B,k,nd]
        h = torch.cat([se_exp, ne, edge_attr], dim=2)                  # [B,k,2nd+ed]
        h = torch.nn.functional.leaky_relu(self.w(h)).view(b, k, self.heads, self.hd)
        score = (h * self.a).sum(-1)                                   # [B,k,heads]
        mask = valid[:, :, None] == 0
        score = score.masked_fill(mask, -1e9)
        alpha = torch.softmax(score, dim=1) * valid[:, :, None]        # zero invalid; all-invalid -> 0
        v = self.wv(ne).view(b, k, self.heads, self.hd)               # [B,k,heads,hd]
        out = (alpha[..., None] * v).sum(1)                           # [B,heads,hd]
        return out.reshape(b, self.heads * self.hd)                   # [B, node_dim]
