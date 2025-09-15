import torch
from torch.nn import Embedding


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, size_of_dicts: list[int], emb_size: int):
        super().__init__()
        self._in_features = len(size_of_dicts)
        self.emb_layers = torch.nn.ModuleList()
        for dict_size in size_of_dicts:
            self.emb_layers.append(
                Embedding(
                    num_embeddings=dict_size, embedding_dim=emb_size, padding_idx=0
                )
            )

    def forward(self, x: torch.Tensor):
        emb_list = []
        for idx, layer in enumerate(self.emb_layers):
            emb_list.append(layer(x[:, idx]))
        x = torch.concatenate(emb_list, dim=1)
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
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:  
                self.activations.append(torch.nn.PReLU())
                self.norms.append(torch.nn.BatchNorm1d(sizes[i+1]))
                if dropout_rate > 0:
                    self.dropout.append(torch.nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        hidden_dim: int = 32,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim

        input_dim = num_experts * expert_output_dim
            
        self.input_dim = input_dim
        
        self.gate_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, num_experts, bias=False),
            torch.nn.Softmax(dim=-1)
        )
        
    def forward(self, expert_outputs: torch.Tensor):
        batch_size = expert_outputs.size(0)
        gate_input = expert_outputs.view(batch_size, -1)
        gate_weights = self.gate_network(gate_input)
    
        expert_outputs_reshaped = expert_outputs.view(batch_size, self.num_experts, self.expert_output_dim)
        
        gate_weights_expanded = gate_weights.unsqueeze(-1)
        weighted_output = torch.sum(
            expert_outputs_reshaped * gate_weights_expanded, dim=1
        )
        return weighted_output, gate_weights