import torch
from torch import nn
from torch.nn import Embedding
from torch_geometric.data import Data


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, size_of_dicts: list[int], emb_size: int):
        super().__init__()
        self._in_features = len(size_of_dicts)
        self.emb_layers = torch.nn.ModuleList()
        for dict_size in size_of_dicts:
            self.emb_layers.append(Embedding(num_embeddings=dict_size, embedding_dim=emb_size, padding_idx=0))

    def forward(self, x: torch.Tensor):
        emb_list = []
        for idx, layer in enumerate(self.emb_layers):
            emb_list.append(layer(x[:, idx]))
        x = torch.cat(emb_list, dim=1)
        return x


class Projector(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        self.out_features = out_features
        if out_features > 32:
            raise ValueError("Wrong out_features in projector layer, should be <= 32")

        sizes = [in_features, 32, 32, out_features]

        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                self.activations.append(torch.nn.PReLU())
                self.norms.append(torch.nn.BatchNorm1d(sizes[i + 1]))
                if dropout_rate > 0:
                    self.dropout.append(torch.nn.Dropout(dropout_rate))

    def forward(self, input: torch.Tensor | Data) -> torch.Tensor:
        if isinstance(input, torch.Tensor):
            x = input
        else:
            x = input.x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
                x = self.norms[i](x)
                if len(self.dropout) > 0:
                    x = self.dropout[i](x)
        return x


class GatingModule(torch.nn.Module):
    def __init__(
        self,
        num_experts: int,
        expert_output_dim: int = 32,
        temp_proj_dim: int = 8,
        use_expert_outputs: bool = True,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim
        self.use_expert_outputs = use_expert_outputs

        input_dim = (num_experts * expert_output_dim if use_expert_outputs else 0) + temp_proj_dim
        self.input_dim = input_dim

        self.gate_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, num_experts, bias=False),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, expert_outputs: torch.Tensor, temp_proj: torch.Tensor):
        batch_size = expert_outputs.size(0)
        if self.use_expert_outputs:
            gate_input = torch.cat([expert_outputs.view(batch_size, -1), temp_proj], dim=1)
        else:
            gate_input = temp_proj
        gate_weights = self.gate_network(gate_input)

        expert_outputs_reshaped = expert_outputs.view(batch_size, self.num_experts, self.expert_output_dim)
        weighted_output = torch.sum(expert_outputs_reshaped * gate_weights.unsqueeze(-1), dim=1)
        return weighted_output, gate_weights


class MLP(nn.Module):
    def __init__(self, hidden_sizes: list[int], act_fn=nn.ReLU, bias: bool = True):
        super().__init__()
        layers = []
        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.extend([nn.Linear(in_size, out_size, bias=bias), act_fn()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CrossNetV2Layer(nn.Module):
    """A single low-rank DCNv2 cross layer."""

    def __init__(self, dim: int, low_rank: int = 16):
        super().__init__()
        self.V = nn.Linear(dim, low_rank, bias=False)
        self.U = nn.Linear(low_rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        x_l = self.V(x)  # (B, low_rank)
        x_l = self.U(x_l)  # (B, dim)
        return x + x0 * (x_l + self.bias)


class CrossNetV2(nn.Module):
    """Stacked CrossNetV2 layers."""

    def __init__(self, dim: int, num_layers: int, low_rank: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([CrossNetV2Layer(dim, low_rank) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for layer in self.layers:
            x = layer(x0, x)
        return x


class DCNv2(nn.Module):
    """
    Deep & Cross Network v2

    structure: 'parallel' or 'stacked'
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int,
        cross_layers: int,
        mlp_sizes: list[int],
        structure: str = "stacked",
        act_fn=nn.PReLU,
        bias: bool = True,
        low_rank: int = 8,
    ):
        super().__init__()

        self.structure = structure
        self.cross_net = CrossNetV2(dim=embedding_dim, num_layers=cross_layers, low_rank=low_rank)

        if structure == "parallel":
            self.mlp = MLP([embedding_dim] + mlp_sizes, act_fn=act_fn, bias=bias)
            self.projection = nn.Linear(embedding_dim + mlp_sizes[-1], out_dim, bias=bias)

        elif structure == "stacked":
            self.mlp = nn.Sequential(MLP([embedding_dim] + mlp_sizes + [out_dim], act_fn=act_fn, bias=bias))
        else:
            raise KeyError(f"No such DCNv2 structure: {structure}")

    def forward(self, x):
        if self.structure == "parallel":
            cross_out = self.cross_net(x)
            deep_out = self.mlp(x)
            return self.projection(torch.cat([cross_out, deep_out], dim=-1))
        elif self.structure == "stacked":
            cross_out = self.cross_net(x)
            return self.mlp(cross_out)
