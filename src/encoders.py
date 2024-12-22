import torch
from torch.nn import Linear, PReLU
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn.norm import GraphNorm


class ThreeLayerGAT(torch.nn.Module):
    """
    A three-layer Graph Attention Network (GAT) model.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        out_dim (int): Output feature dimension.
        edge_dim (int): Edge feature dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, edge_dim: int, num_heads: int
    ):
        super().__init__()
        self._edgeMLP = Linear(edge_dim, hidden_dim)

        self._gat1 = GATConv(
            in_dim, hidden_dim, edge_dim=hidden_dim, heads=num_heads, concat=True
        )
        self._gat2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            edge_dim=hidden_dim,
            heads=num_heads,
            concat=True,
        )
        self._gat3 = GATConv(
            hidden_dim * num_heads,
            out_dim,
            edge_dim=hidden_dim,
            heads=num_heads,
            concat=False,
        )
        self._norm0 = GraphNorm(hidden_dim)
        self._norm1 = GraphNorm(hidden_dim * num_heads)
        self._norm2 = GraphNorm(hidden_dim * num_heads)
        self._norm3 = GraphNorm(out_dim)

        self._act0 = PReLU()
        self._act1 = PReLU()
        self._act2 = PReLU()
        self._act3 = PReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor | None,
    ):
        edge_attr = self._act0(self._norm0(self._edgeMLP(edge_attr), batch_index))

        x = self._gat1(x, edge_index, edge_attr)
        x = self._norm1(x, batch_index)
        x = self._act1(x)

        x = self._gat2(x, edge_index, edge_attr)
        x = self._norm2(x, batch_index)
        x = self._act2(x)

        x = self._gat3(x, edge_index, edge_attr)
        x = self._norm3(x, batch_index)
        x = self._act3(x)

        return x
