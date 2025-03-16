from abc import ABC, abstractmethod

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import ModuleList, PReLU
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv as GATConv
from torch_geometric.nn.norm import GraphNorm

from generic import EmbeddingLayer


class BaseEncoder(ABC, torch.nn.Module):
    """
    Abstract base class for all encoders.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int,
        projector: DictConfig,
        global_pool: DictConfig,
        dropout_rate: float,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor | None,
    ):
        pass


class GATEncoder(BaseEncoder):
    """
    A configurable Graph Attention Network (GAT) model with a variable number of layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int,
        projector: DictConfig,
        global_pool: DictConfig,
        num_layers: int = 3,
        dropout_rate: float = 0.5,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            node_size_of_dicts=node_size_of_dicts,
            edge_size_of_dicts=edge_size_of_dicts,
            emb_size=emb_size,
            projector=projector,
            global_pool=global_pool,
            dropout_rate=dropout_rate,
        )
        # Output features and dimensions
        node_dim = len(node_size_of_dicts)
        edge_dim = len(edge_size_of_dicts)
        concat_out_size = (
            node_dim * emb_size + (num_layers - 1) * hidden_dim * num_heads + hidden_dim
        )
        self.out_features = projector.out_features
        self.num_layers = num_layers

        # Embedding layers
        self.node_emb_layer = EmbeddingLayer(node_size_of_dicts, emb_size)
        self.edge_emb_layer = EmbeddingLayer(edge_size_of_dicts, emb_size)
        self.emb_pool = instantiate(global_pool)

        # GAT layers
        self.gat_layers = ModuleList()
        self.norm_layers = ModuleList()
        self.act_layers = ModuleList()
        self.pool_layers = ModuleList()
        input_dim = emb_size * node_dim
        self.dropout_layers = ModuleList()

        for i in range(self.num_layers):
            is_last_layer = i == self.num_layers - 1
            output_dim = out_dim if is_last_layer else hidden_dim
            concat = not is_last_layer
            self.gat_layers.append(
                GATConv(
                    input_dim,
                    output_dim,
                    edge_dim=emb_size * edge_dim,
                    heads=num_heads,
                    concat=concat,
                )
            )
            norm_dim = output_dim if not concat else output_dim * num_heads
            self.norm_layers.append(GraphNorm(norm_dim))
            self.act_layers.append(PReLU())
            self.pool_layers.append(instantiate(global_pool))
            self.dropout_layers.append(torch.nn.Dropout(dropout_rate))
            input_dim = norm_dim

        # Projector layer
        self.projector_layer: torch.nn.Module = instantiate(
            projector, in_features=concat_out_size
        )

    def forward(self, graph: Data):  # type: ignore
        # Unpack graph data
        x, edge_index, edge_attr, batch_index = (
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
        )
        embeddings = []

        x = self.node_emb_layer(x)
        edge_attr = self.edge_emb_layer(edge_attr)
        embeddings.append(self.emb_pool(x, batch_index))

        for i in range(self.num_layers):
            x = self.gat_layers[i](x, edge_index, edge_attr)
            x = self.norm_layers[i](x, batch_index)
            x = self.act_layers[i](x)
            x = self.dropout_layers[i](x)
            embeddings.append(self.pool_layers[i](x, batch_index))
        x = torch.cat(embeddings, dim=-1)
        x = self.projector_layer(x)

        if self.out_features == 1:
            x = x.squeeze()
        return x


class DoubleGraphPredictor(torch.nn.Module):
    def __init__(self, encoder: DictConfig, projector: DictConfig):
        super().__init__()
        self.encoder_1: BaseEncoder = instantiate(encoder, _recursive_=False)
        self.encoder_2: BaseEncoder = instantiate(encoder, _recursive_=False)
        self.final_projector_layer = instantiate(
            projector,
            in_features=(self.encoder_1.out_features + self.encoder_2.out_features),
        )

    def forward(self, graph_1: Data, graph_2: Data):
        x_1 = self.encoder_1(graph_1)
        x_2 = self.encoder_2(graph_2)
        x = torch.cat([x_1, x_2], dim=-1)
        x = self.final_projector_layer(x)
        return x
