import torch
from torch.nn import PReLU, BatchNorm1d
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn.norm import GraphNorm
from generic import EmbeddingLayer


# TODO: Create BaseEncoder class as parent class for all encoders
class ThreeLayerGAT(torch.nn.Module):
    """
    A three-layer Graph Attention Network (GAT) model.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        out_dim (int): Output feature dimension. Typically equal to hidden_dim.
        edge_dim (int): Edge feature dimension.
        num_heads (int): Number of attention heads.
        node_size_of_dicts (list[int]): Size of dicts for node embedding layers.
        edge_size_of_dicts (list[int]): Size of dicts for edge embedding layers.
        emb_size (int): Dimensionality of embedding for each feature.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        edge_dim: int,
        num_heads: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.node_emb_layer = EmbeddingLayer(node_size_of_dicts, self.emb_size)
        self.edge_emb_layer = EmbeddingLayer(edge_size_of_dicts, self.emb_size)
        self._gat1 = GATConv(
            self.emb_size * in_dim,
            hidden_dim,
            edge_dim=self.emb_size * edge_dim,
            heads=num_heads,
            concat=True,
        )
        self._gat2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            edge_dim=self.emb_size * edge_dim,
            heads=num_heads,
            concat=True,
        )
        self._gat3 = GATConv(
            hidden_dim * num_heads,
            out_dim,
            edge_dim=self.emb_size * edge_dim,
            heads=num_heads,
            concat=False,
        )
        self._norm_node = GraphNorm(hidden_dim)
        self._norm_edge = BatchNorm1d(hidden_dim)
        self._norm1 = GraphNorm(hidden_dim * num_heads)
        self._norm2 = GraphNorm(hidden_dim * num_heads)
        self._norm3 = GraphNorm(out_dim)

        self._act_node = PReLU()
        self._act_edge = PReLU()
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
        x = self.node_emb_layer(x)
        edge_attr = self.edge_emb_layer(edge_attr)

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
