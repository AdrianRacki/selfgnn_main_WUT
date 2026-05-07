from abc import ABC, abstractmethod

from torch import nn
from torch.nn import Linear, ModuleList, PReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, TransformerConv
from torch_geometric.nn.norm import GraphNorm

from model.generic import DCNv2


class BaseEncoder(ABC, nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        node_input_dim: int,
        edge_input_dim: int,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.gnn_layers = self._get_gnn_layers()
        self.norm_layers = ModuleList()
        self.act_layers = ModuleList()

        for i in range(self.num_layers):
            layer_input_dim = self._get_layer_input_dim(i)
            self.norm_layers.append(GraphNorm(layer_input_dim))
            self.act_layers.append(PReLU())

    @abstractmethod
    def _get_gnn_layers(self) -> ModuleList:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

    def _get_layer_input_dim(self, i: int) -> int:
        return self.output_dim

    def forward(self, graph: Data):  # type: ignore
        # Unpack graph data
        x, edge_index, edge_attr, batch_index, _ = (
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            graph.g,
        )

        for i in range(self.num_layers):
            fx = x
            x = self.gnn_layers[i](x, edge_index, edge_attr)
            x = self.norm_layers[i](x, batch_index)
            x = self.act_layers[i](x)
            x = x + fx if i > 0 else x
        return x


class GATEncoder(BaseEncoder):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        node_input_dim: int,
        edge_input_dim: int,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
    ):
        self.num_heads = num_heads
        super().__init__(
            hidden_dim=hidden_dim,
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * self.num_heads

    def _get_gnn_layers(self) -> ModuleList:
        gnn_layers = ModuleList()
        input_dim = self.input_dim

        for i in range(self.num_layers):
            gnn_layers.append(
                TransformerConv(
                    input_dim,
                    self.hidden_dim,
                    edge_dim=self.edge_input_dim,
                    heads=self.num_heads,
                    concat=True,
                    beta=False,
                    dropout=self.dropout_rate,
                )
            )
            input_dim = self.hidden_dim * self.num_heads

        return gnn_layers


class GINEncoder(BaseEncoder):
    def __init__(
        self,
        hidden_dim: int,
        node_input_dim: int,
        edge_input_dim: int,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        internal_model: str = "MLP",  # "MLP" or "CrossNetV2"
        cross_layers: int = 1,
    ):
        self.internal_model = internal_model
        self.cross_layers = cross_layers
        super().__init__(
            hidden_dim=hidden_dim,
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def _create_mlp(self, input_dim: int, output_dim: int) -> Sequential:
        return Sequential(
            Linear(input_dim, output_dim),
            PReLU(),
            Linear(output_dim, output_dim),
            PReLU(),
        )

    def _create_crossnet(self, input_dim: int, output_dim: int) -> DCNv2:
        return DCNv2(
            embedding_dim=input_dim,
            out_dim=output_dim,
            cross_layers=self.cross_layers,
            mlp_sizes=[16, 16],
            structure="parallel",
        )

    def _get_gnn_layers(self) -> ModuleList:
        gnn_layers = ModuleList()
        input_dim = self.input_dim
        for i in range(self.num_layers):
            if self.internal_model == "CrossNetV2":
                mlp = self._create_crossnet(input_dim, self.hidden_dim)
            elif self.internal_model == "MLP":
                mlp = self._create_mlp(input_dim, self.hidden_dim)
            else:
                raise ValueError(f"Unknown internal_model: {self.internal_model}")

            mlp.in_channels = input_dim
            gnn_layers.append(
                GINEConv(
                    nn=mlp,
                    train_eps=True,
                    edge_dim=self.edge_input_dim,
                )
            )
            input_dim = self.hidden_dim

        return gnn_layers
