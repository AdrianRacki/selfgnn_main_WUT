import torch
import torch_geometric.nn
from torch import nn
from torch_geometric.data import Data


class GraphPooling(nn.Module):
    def __init__(self, pooling_layer: nn.Module):
        super().__init__()
        self.pooling_layer = pooling_layer

    def forward(self, graph: Data) -> torch.Tensor:
        return self.pooling_layer(graph.x, graph.batch)


class IDACPooling(torch.nn.Module):
    def __init__(self, num_mols: int = 3):
        super().__init__()
        self.num_mols = num_mols
        self.pooling_layer = torch_geometric.nn.aggr.SoftmaxAggregation(t=2, learn=True)

    @property
    def output_multiplier(self) -> int:
        return self.num_mols

    def forward(self, graph: Data) -> torch.Tensor:
        B = int(graph.batch.max().item()) + 1
        mol_id = graph.batch * self.num_mols + graph.map  # unique ID per (batch_item, mol)
        pooled = self.pooling_layer(graph.x, mol_id, dim_size=B * self.num_mols)  # [B*num_mols, features]
        return pooled.view(B, -1)  # [B, num_mols * features]
