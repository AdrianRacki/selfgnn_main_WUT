import torch
from torch_geometric.data import Data


class IDACProjector(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        dropout_rate: float = 0.3,
        append_global_features: bool = True,
        global_features_size: int = 45,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        self.out_features = out_features
        self.append_global_features = append_global_features
        temperature_size = 8
        self.temperature_projector = torch.nn.Linear(1, temperature_size, bias=False)
        in_features = in_features + temperature_size + global_features_size if append_global_features else in_features + temperature_size
        self.input_batchnorm = torch.nn.BatchNorm1d(in_features)
        sizes = [in_features, 64, 32, out_features]

        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                self.activations.append(torch.nn.PReLU())
                self.norms.append(torch.nn.BatchNorm1d(sizes[i + 1]))
                if dropout_rate > 0:
                    self.dropout.append(torch.nn.Dropout(dropout_rate))

    def forward(self, graph: torch.Tensor | Data) -> torch.Tensor:
        temp = graph.temperature.float().view(-1, 1)  # [B] → [B, 1]
        x = torch.cat([graph.x, self.temperature_projector(temp)], dim=-1)
        if self.append_global_features:
            B = x.shape[0]
            g = graph.g.view(B, -1)  # [B*num_mols, n_feats] → [B, num_mols*n_feats]
            x = torch.cat([x, g], dim=-1)
        x = self.input_batchnorm(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
                x = self.norms[i](x)
                if len(self.dropout) > 0:
                    x = self.dropout[i](x)
        return x
