from collections.abc import Callable

import torch
from torch import nn
from torch_geometric.data import Data

from utils.graph_tools import split_graph_by_map


class GraphPooling(nn.Module):
    def __init__(self, pooling_layer: nn.Module):
        super().__init__()
        self.pooling_layer = pooling_layer

    def forward(self, graph: Data) -> torch.Tensor:
        return self.pooling_layer(graph.x, graph.batch)


class IDACPooling(torch.nn.Module):
    def __init__(self, pooling_layer: Callable[[], nn.Module], num_mols: int = 3, separate_poolers: bool = True):
        super().__init__()
        self.num_mols = num_mols
        self.separate_poolers = separate_poolers
        if separate_poolers:
            self.pooling_layers = nn.ModuleList([pooling_layer() for _ in range(num_mols)])
        else:
            self.pooling_layer = pooling_layer()

    @property
    def output_multiplier(self) -> int:
        return self.num_mols

    def forward(self, graph: Data) -> torch.Tensor:
        B = int(graph.batch.max().item()) + 1
        if self.separate_poolers:
            parts = [
                self.pooling_layers[mol_idx](x_mol, batch_mol, dim_size=B)  # [B, features]
                for mol_idx, (x_mol, batch_mol) in enumerate(split_graph_by_map(graph))
            ]
            return torch.cat(parts, dim=-1)  # [B, num_mols * features]
        mol_id = graph.batch * self.num_mols + graph.map  # unique ID per (batch_item, mol)
        pooled = self.pooling_layer(graph.x, mol_id, dim_size=B * self.num_mols)  # [B*num_mols, features]
        return pooled.view(B, -1)  # [B, num_mols * features]


class IDACTransformerPooling(IDACPooling):
    """Per-molecule softmax pooling returning stacked token embeddings [B, num_mols, features]."""

    def forward(self, graph: Data) -> torch.Tensor:
        B = int(graph.batch.max().item()) + 1
        if self.separate_poolers:
            parts = [
                self.pooling_layers[mol_idx](x_mol, batch_mol, dim_size=B)
                for mol_idx, (x_mol, batch_mol) in enumerate(split_graph_by_map(graph))
            ]
            return torch.stack(parts, dim=1)  # [B, num_mols, features]
        mol_id = graph.batch * self.num_mols + graph.map
        pooled = self.pooling_layer(graph.x, mol_id, dim_size=B * self.num_mols)
        return pooled.view(B, self.num_mols, -1)  # [B, num_mols, features]
