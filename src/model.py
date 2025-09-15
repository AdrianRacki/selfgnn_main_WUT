from abc import ABC, abstractmethod

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import ModuleList, PReLU, Linear, Sequential, Dropout
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, GCNConv, GINConv, ARMAConv, EGConv, NNConv
from torch_geometric.nn.norm import GraphNorm

from generic import EmbeddingLayer, GatingModule, Projector

DEFAULT_GLOBAL_POOL = DictConfig({
    "_target_": "torch_geometric.nn.aggr.SoftmaxAggregation",
    "t": 2,
    "learn": True})

class GraphModelWrapper(torch.nn.Module):
    def __init__(self, experts: list[DictConfig]) -> None:
        super().__init__()
        
        self.experts = ModuleList([instantiate(expert) for expert in experts])
        self.gating_module = GatingModule(
            num_experts=len(self.experts),
            expert_output_dim=self.experts[0].out_features,
        )
        self.projector_layer = Projector(in_features=self.experts[0].out_features)
        
    def forward(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = [expert(graph) for expert in self.experts]
        expert_outputs = torch.cat(expert_outputs, dim=1)
        weighted_output, gate_weights = self.gating_module(expert_outputs)
        output = self.projector_layer(weighted_output).squeeze()
        return output, gate_weights

class GlobalFeaturesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Projector(
            in_features=15,
            out_features=32)

    def forward(self, graph: Data) -> torch.Tensor:
        x = graph.g
        x = self.mlp(x)
        return x
    
class BaseEncoder(ABC, torch.nn.Module):
    def __init__( 
        self,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        global_pool: DictConfig = DEFAULT_GLOBAL_POOL,
        emb_size: int = 4,
        num_layers: int = 3,
        out_features: int = 32,
    ):
        super().__init__()
        
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.out_features = out_features
        
        self.node_dim = len(node_size_of_dicts)
        self.edge_dim = len(edge_size_of_dicts)

        self.node_emb_layer = EmbeddingLayer(node_size_of_dicts, emb_size)
        self.edge_emb_layer = EmbeddingLayer(edge_size_of_dicts, emb_size)
        self.node_pool = instantiate(global_pool)
        self.emb_pool = instantiate(global_pool)
        
        self.gnn_layers = self._get_gnn_layers()
        self.norm_layers = ModuleList()
        self.act_layers = ModuleList()
        
        gnn_output_dim = self._get_gnn_output_dim()
        concat_out_size = self.node_dim * emb_size + gnn_output_dim
        
        for i in range(self.num_layers):
            layer_input_dim = self._get_layer_input_dim(i)
            self.norm_layers.append(GraphNorm(layer_input_dim))
            self.act_layers.append(PReLU())
        
        self.projector_layer: torch.nn.Module = Projector(
            in_features=concat_out_size,
            out_features=self.out_features,
        )

    @abstractmethod
    def _get_gnn_layers(self) -> ModuleList:
        pass
    
    @abstractmethod
    def _get_gnn_output_dim(self) -> int:
        pass
    
    @abstractmethod
    def _get_layer_input_dim(self, layer_idx: int) -> int:
        pass
    
    @property
    @abstractmethod
    def use_edge_weights(self) -> bool:
        pass

    def forward(self, graph: Data):  # type: ignore
        # Unpack graph data
        x, edge_index, edge_attr, batch_index, _ = (
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            graph.g,
        )
        embeddings = []

        x = self.node_emb_layer(x)
        edge_attr = self.edge_emb_layer(edge_attr)
        embeddings.append(self.node_pool(x, batch_index))

        for i in range(self.num_layers):
            fx = x
            if self.use_edge_weights:
                x = self.gnn_layers[i](x, edge_index, edge_attr)
            else:
                x = self.gnn_layers[i](x, edge_index)
            x = self.norm_layers[i](x, batch_index)
            x = self.act_layers[i](x)
            x = x + fx if i > 0 else x  # Residual connection
        
        x = self.emb_pool(x, batch_index)
        x = torch.cat(embeddings + [x], dim=-1)
        x = self.projector_layer(x)
        return x


class GATEncoder(BaseEncoder):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int = 4,
        global_pool: DictConfig = DEFAULT_GLOBAL_POOL,
        num_layers: int = 3,
        out_features: int = 32,
        dropout_rate: float = 0.3,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        super().__init__(
            node_size_of_dicts=node_size_of_dicts,
            edge_size_of_dicts=edge_size_of_dicts,
            emb_size=emb_size,
            global_pool=global_pool,
            num_layers=num_layers,
            out_features=out_features,
        )

    def _get_gnn_layers(self) -> ModuleList:
        gnn_layers = ModuleList()
        input_dim = self.emb_size * self.node_dim
        
        for i in range(self.num_layers):
            output_dim = self.hidden_dim
            gnn_layers.append(
                TransformerConv(
                    input_dim,
                    output_dim,
                    edge_dim=self.emb_size * self.edge_dim,
                    heads=self.num_heads,
                    concat=True,
                    beta=False,
                    dropout=self.dropout_rate,
                )
            )
            input_dim = output_dim * self.num_heads
        
        return gnn_layers
    
    def _get_gnn_output_dim(self) -> int:
        return self.hidden_dim * self.num_heads
    
    def _get_layer_input_dim(self, layer_idx: int) -> int:
        return self.hidden_dim * self.num_heads
    
    @property
    def use_edge_weights(self) -> bool:
        return True


class GCNEncoder(BaseEncoder):
    def __init__(
        self,
        hidden_dim: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int = 4,
        global_pool: DictConfig = DEFAULT_GLOBAL_POOL,
        num_layers: int = 3,
        out_features: int = 32,
        dropout_rate: float = 0.3,
    ):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        super().__init__(
            node_size_of_dicts=node_size_of_dicts,
            edge_size_of_dicts=edge_size_of_dicts,
            emb_size=emb_size,
            global_pool=global_pool,
            num_layers=num_layers,
            out_features=out_features,
        )

    def _get_gnn_layers(self) -> ModuleList:
        gnn_layers = ModuleList()
        input_dim = self.emb_size * self.node_dim
        
        for i in range(self.num_layers):
            output_dim = self.hidden_dim
            gnn_layers.append(
                GCNConv(
                    input_dim,
                    output_dim,
                    improved=True,
                )
            )
            input_dim = output_dim
        
        return gnn_layers
    
    def _get_gnn_output_dim(self) -> int:
        return self.hidden_dim
    
    def _get_layer_input_dim(self, layer_idx: int) -> int:
        return self.hidden_dim
    
    @property
    def use_edge_weights(self) -> bool:
        return False


class GINEncoder(BaseEncoder):
    def __init__(
        self,
        hidden_dim: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int = 4,
        global_pool: DictConfig = DEFAULT_GLOBAL_POOL,
        num_layers: int = 3,
        out_features: int = 32,
        dropout_rate: float = 0.3,
    ):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        super().__init__(
            node_size_of_dicts=node_size_of_dicts,
            edge_size_of_dicts=edge_size_of_dicts,
            emb_size=emb_size,
            global_pool=global_pool,
            num_layers=num_layers,
            out_features=out_features,
        )

    def _create_mlp(self, input_dim: int, output_dim: int) -> Sequential:
        return Sequential(
            Linear(input_dim, output_dim),
            PReLU(),
            Linear(output_dim, output_dim),
            PReLU()
        )

    def _get_gnn_layers(self) -> ModuleList:
        gnn_layers = ModuleList()
        input_dim = self.emb_size * self.node_dim
        
        for i in range(self.num_layers):
            output_dim = self.hidden_dim
            
            mlp = self._create_mlp(input_dim, output_dim)
            
            gnn_layers.append(
                GINConv(
                    nn=mlp,
                    train_eps=True,
                )
            )
            input_dim = output_dim
        
        return gnn_layers
    
    def _get_gnn_output_dim(self) -> int:
        return self.hidden_dim
    
    def _get_layer_input_dim(self, layer_idx: int) -> int:
        return self.hidden_dim
    
    @property
    def use_edge_weights(self) -> bool:
        return False


class ARMAEncoder(BaseEncoder):
    def __init__(
        self,
        hidden_dim: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int = 4,
        global_pool: DictConfig = DEFAULT_GLOBAL_POOL,
        num_layers: int = 3,
        out_features: int = 32,
        dropout_rate: float = 0.3,
    ):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        super().__init__(
            node_size_of_dicts=node_size_of_dicts,
            edge_size_of_dicts=edge_size_of_dicts,
            emb_size=emb_size,
            global_pool=global_pool,
            num_layers=num_layers,
            out_features=out_features,
        )

    def _get_gnn_layers(self) -> ModuleList:
        gnn_layers = ModuleList()
        input_dim = self.emb_size * self.node_dim
        
        for i in range(self.num_layers):
            output_dim = self.hidden_dim
            gnn_layers.append(
                ARMAConv(
                    input_dim,
                    output_dim,
                )
            )
            input_dim = output_dim
        
        return gnn_layers
    
    def _get_gnn_output_dim(self) -> int:
        return self.hidden_dim
    
    def _get_layer_input_dim(self, layer_idx: int) -> int:
        return self.hidden_dim
    
    @property
    def use_edge_weights(self) -> bool:
        return False  # ARMAConv doesn't use edge attributes


class EGConvEncoder(BaseEncoder):
    def __init__(
        self,
        hidden_dim: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int = 4,
        global_pool: DictConfig = DEFAULT_GLOBAL_POOL,
        num_layers: int = 3,
        out_features: int = 32,
        dropout_rate: float = 0.3,
        aggregators: list[str] | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.aggregators = aggregators if aggregators is not None else ["mean", "symnorm", "max", "min"]
        
        super().__init__(
            node_size_of_dicts=node_size_of_dicts,
            edge_size_of_dicts=edge_size_of_dicts,
            emb_size=emb_size,
            global_pool=global_pool,
            num_layers=num_layers,
            out_features=out_features,
        )

    def _get_gnn_layers(self) -> ModuleList:
        gnn_layers = ModuleList()
        input_dim = self.emb_size * self.node_dim
        
        for i in range(self.num_layers):
            output_dim = self.hidden_dim
            gnn_layers.append(
                EGConv(
                    input_dim,
                    output_dim,
                    aggregators=self.aggregators,
                )
            )
            input_dim = output_dim
        
        return gnn_layers
    
    def _get_gnn_output_dim(self) -> int:
        return self.hidden_dim 
    
    def _get_layer_input_dim(self, layer_idx: int) -> int:
        return self.hidden_dim 
    
    @property
    def use_edge_weights(self) -> bool:
        return False


class NNConvEncoder(BaseEncoder):
    """
    A configurable Neural Network Convolution (NNConv) model with a variable number of layers.
    Uses NNConv with a simple two-layer MLP similar to GINEncoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int = 4,
        global_pool: DictConfig = DEFAULT_GLOBAL_POOL,
        num_layers: int = 3,
        out_features: int = 32,
        dropout_rate: float = 0.3,
    ):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        super().__init__(
            node_size_of_dicts=node_size_of_dicts,
            edge_size_of_dicts=edge_size_of_dicts,
            emb_size=emb_size,
            global_pool=global_pool,
            num_layers=num_layers,
            out_features=out_features,
        )

    def _create_mlp(self, input_dim: int, output_dim: int, edge_dim: int) -> Sequential:
        hidden_dim = max(input_dim, output_dim) 
        return Sequential(
            Linear(edge_dim, hidden_dim),
            PReLU(),
            Linear(hidden_dim, input_dim * output_dim),
            PReLU()
        )

    def _get_gnn_layers(self) -> ModuleList:
        gnn_layers = ModuleList()
        input_dim = self.emb_size * self.node_dim
        edge_dim = self.emb_size * self.edge_dim
        
        for i in range(self.num_layers):
            output_dim = self.hidden_dim
            
            mlp = self._create_mlp(input_dim, output_dim, edge_dim)
            
            gnn_layers.append(
                NNConv(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    nn=mlp,
                )
            )
            input_dim = output_dim
        
        return gnn_layers
    
    def _get_gnn_output_dim(self) -> int:
        return self.hidden_dim
    
    def _get_layer_input_dim(self, layer_idx: int) -> int:
        return self.hidden_dim
    
    @property
    def use_edge_weights(self) -> bool:
        return True


class DMPNN(torch.nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Initialization layer
        self.W_int = Sequential(
            Linear(self.node_dim + self.edge_dim, self.hidden_dim),
            PReLU(),
            Dropout(0.3),
        )
        self.W_h = Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            PReLU(),
        )
        self.update_dropout = Dropout(0.3)
        
        # Finalization layers for vertex and edge
        self.W_eo = Linear(self.edge_dim + self.hidden_dim, self.hidden_dim)
        self.tau = PReLU()
        self.dropout = Dropout(0.3)

    def initialize(self, x, edge_index, edge_attr) -> torch.Tensor:
        return self.W_int(torch.cat([x[edge_index[0]], edge_attr], dim=1))

    def message(self, H: torch.Tensor, x, edge_index, rev_edge_index) -> torch.Tensor:
        index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1]) # type: ignore
        M_all = torch.zeros(len(x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )[edge_index[0]] # type: ignore
        M_rev = H[rev_edge_index]

        return M_all - M_rev
    
    def update(self, M: torch.Tensor, H_0: torch.Tensor) -> torch.Tensor:
        H_t = self.W_h(M) + H_0
        return self.update_dropout(H_t)


    def edge_finalize(self, H: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Finalize message passing for edge embeddings by concatenating the final hidden 
        directed edges H and the original edge features E.
        
        Parameters
        ----------
        H : torch.Tensor
            a tensor containing the hidden state for each edge
        E : torch.Tensor
            a tensor containing the original edge features
            
        Returns
        -------
        torch.Tensor
            a tensor containing the final hidden representations
        """
        H = self.W_eo(torch.cat((E, H), dim=1))
        H = self.tau(H)
        H = self.dropout(H)
        return H

    def forward(self, x, edge_index, edge_attr, rev_edge_index) ->torch.Tensor:
        H_0 = self.initialize(x, edge_index, edge_attr)
        H_t = H_0
        for _ in range(self.num_layers):
            M = self.message(H_t, x, edge_index, rev_edge_index)
            H_t = self.update(M, H_0)

        H_e = self.edge_finalize(H_t, edge_attr)
        return H_e

class DMPNNEncoder(BaseEncoder):
    def __init__(
        self,
        hidden_dim: int,
        node_size_of_dicts: list[int],
        edge_size_of_dicts: list[int],
        emb_size: int = 4,
        global_pool: DictConfig = DEFAULT_GLOBAL_POOL,
        num_layers: int = 3,
        out_features: int = 32,
    ):
        self.hidden_dim = hidden_dim
        super().__init__(
            node_size_of_dicts=node_size_of_dicts,
            edge_size_of_dicts=edge_size_of_dicts,
            emb_size=emb_size,
            global_pool=global_pool,
            num_layers=num_layers,
            out_features=out_features,
        )
        self.projector_layer: torch.nn.Module = Projector(
            in_features=self.hidden_dim,
            out_features=self.out_features,
        )
        self.norm_layers = ModuleList()
        self.act_layers = ModuleList()
    def _get_gnn_layers(self) -> ModuleList:
        gnn_layers = ModuleList()
        gnn_layers.append(
            DMPNN(
                node_dim=self.emb_size * self.node_dim,
                edge_dim=self.emb_size * self.edge_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
        )
        return gnn_layers
    
    def _get_gnn_output_dim(self) -> int:
        return self.out_features

    def _get_layer_input_dim(self, layer_idx: int) -> int:
        return self.out_features

    def forward(self, graph: Data):  # type: ignore
        # Unpack graph data
        x, edge_index, edge_attr, batch_index, _, rev_edge_index = (
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
            graph.g,
            graph.rev_edge_index
        )

        x = self.node_emb_layer(x)
        edge_attr = self.edge_emb_layer(edge_attr)

        x = self.gnn_layers[0](x, edge_index, edge_attr, rev_edge_index)
        edge_batch = batch_index[edge_index[0]] # type: ignore
        x = self.emb_pool(x, edge_batch)
        x = self.projector_layer(x)
        return x
        
    @property
    def use_edge_weights(self) -> bool:
        return True 