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
    def __init__(self, num_mols: int = 3, channels: int = 1, separate_poolers: bool = True):
        super().__init__()
        self.num_mols = num_mols
        self.separate_poolers = separate_poolers
        if separate_poolers:
            self.pooling_layers = nn.ModuleList(
                [torch_geometric.nn.aggr.SoftmaxAggregation(t=2, learn=True, channels=channels) for _ in range(num_mols)]
            )
        else:
            self.pooling_layer = torch_geometric.nn.aggr.SoftmaxAggregation(t=2, learn=True, channels=channels)

    @property
    def output_multiplier(self) -> int:
        return self.num_mols

    def forward(self, graph: Data) -> torch.Tensor:
        B = int(graph.batch.max().item()) + 1
        if self.separate_poolers:
            parts = []
            for mol_idx in range(self.num_mols):
                mask = graph.map == mol_idx
                x_mol = graph.x[mask]
                batch_mol = graph.batch[mask]
                parts.append(self.pooling_layers[mol_idx](x_mol, batch_mol, dim_size=B))  # [B, features]
            return torch.cat(parts, dim=-1)  # [B, num_mols * features]
        mol_id = graph.batch * self.num_mols + graph.map  # unique ID per (batch_item, mol)
        pooled = self.pooling_layer(graph.x, mol_id, dim_size=B * self.num_mols)  # [B*num_mols, features]
        return pooled.view(B, -1)  # [B, num_mols * features]
