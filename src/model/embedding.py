import torch
from torch import nn
from torch_geometric.data import Data

from model.generic import EmbeddingLayer
from utils.graph_tools import split_global_to_mols


class GraphEmbeddingLayer(nn.Module):
    def __init__(
        self,
        node_feature_sizes: list[int],
        edge_feature_sizes: list[int],
        emb_size: int = 6,
        global_features_module: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.emb_size = emb_size
        self.node_dim = len(node_feature_sizes)
        self.edge_dim = len(edge_feature_sizes)

        self.node_emb_layer = EmbeddingLayer(node_feature_sizes, emb_size)
        self.edge_emb_layer = EmbeddingLayer(edge_feature_sizes, emb_size)
        self.global_features_module = global_features_module

    def forward(self, graph: Data) -> Data:
        graph.x = self.node_emb_layer(graph.x)
        graph.edge_attr = self.edge_emb_layer(graph.edge_attr)
        if graph.g.size(0) > 1:
            graph.g = torch.cat(
                [self.global_features_module(g) for g in split_global_to_mols(graph)],
                dim=0,
            )
        else:
            graph.g = self.global_features_module(graph.g)
        return graph

    @property
    def node_output_dim(self) -> int:
        return self.node_emb_layer.output_dim

    @property
    def edge_output_dim(self) -> int:
        return self.edge_emb_layer.output_dim
