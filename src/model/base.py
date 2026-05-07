import torch
from torch import nn
from torch_geometric.data import Data

OUTPUT_KEY = "output"


class BaseModel(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Module,
        encoder_layer: nn.Module,
        pooling_layer: nn.Module,
        projector_layer: nn.Module,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder_layer = encoder_layer(
            node_input_dim=self.embedding_layer.node_output_dim,
            edge_input_dim=self.embedding_layer.edge_output_dim,
        )
        self.pooling_layer = pooling_layer(channels=self.encoder_layer.output_dim)
        pooling_multiplier = getattr(self.pooling_layer, "output_multiplier", 1)
        self.projector_layer = projector_layer(in_features=self.encoder_layer.output_dim * pooling_multiplier)

    def forward(self, graph: Data) -> dict[str, torch.Tensor]:
        graph = self.embedding_layer(graph)
        graph.x = self.encoder_layer(graph)
        graph.x = self.pooling_layer(graph)
        output = self.projector_layer(graph).squeeze(-1)
        return {OUTPUT_KEY: output}
