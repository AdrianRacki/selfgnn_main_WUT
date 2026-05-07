import torch
from torch import nn
from torch.nn import Linear, ModuleList, PReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, TransformerConv
from torch_geometric.nn.norm import GraphNorm

from model.generic import DCNv2


class GatedEncoder(nn.Module):
    """
    A gated mixture-of-experts encoder that wraps multiple GNN expert stacks
    with a soft gating mechanism. All experts share the upstream embedding tables.

    When `shared_expert=True`, expert 0 always contributes to the output (not gated),
    and only experts 1..N are combined via gating. The final output is:
        output = expert_0_output + gated_combination(expert_1..N)

    When `shared_expert=False`, all experts are combined via soft gating.
    """

    def __init__(
        self,
        hidden_dim: int,
        node_input_dim: int,
        edge_input_dim: int,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        n_experts: int = 3,
        encoder_type: str = "GIN",
        shared_expert: bool = False,
        # GIN-specific
        internal_model: str = "MLP",
        cross_layers: int = 1,
        # GAT-specific
        num_heads: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.n_experts = n_experts
        self.encoder_type = encoder_type
        self.shared_expert = shared_expert
        self.internal_model = internal_model
        self.cross_layers = cross_layers
        self.num_heads = num_heads

        # Build expert GNN stacks
        self.experts = ModuleList()
        for _ in range(n_experts):
            self.experts.append(self._build_expert())

        expert_out_dim = self._expert_output_dim()

        # Shared norm/activation per layer (shared across experts)
        self.norm_layers = ModuleList()
        self.act_layers = ModuleList()
        for _ in range(num_layers):
            self.norm_layers.append(GraphNorm(expert_out_dim))
            self.act_layers.append(PReLU())

        # Gating network: takes node features and produces per-expert weights
        n_gated = n_experts if not shared_expert else n_experts - 1
        self.gate_network = Sequential(
            Linear(node_input_dim, n_gated),
            nn.Softmax(dim=-1),
        )

    def _expert_output_dim(self) -> int:
        if self.encoder_type == "GAT":
            return self.hidden_dim * self.num_heads
        return self.hidden_dim

    @property
    def output_dim(self) -> int:
        return self._expert_output_dim()

    def _build_expert(self) -> ModuleList:
        """Build a stack of GNN layers for one expert."""
        layers = ModuleList()
        input_dim = self.input_dim

        for _ in range(self.num_layers):
            if self.encoder_type == "GAT":
                layers.append(
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
            elif self.encoder_type == "GIN":
                if self.internal_model == "CrossNetV2":
                    mlp = DCNv2(
                        embedding_dim=input_dim,
                        out_dim=self.hidden_dim,
                        cross_layers=self.cross_layers,
                        mlp_sizes=[16, 16],
                        structure="parallel",
                    )
                elif self.internal_model == "MLP":
                    mlp = Sequential(
                        Linear(input_dim, self.hidden_dim),
                        PReLU(),
                        Linear(self.hidden_dim, self.hidden_dim),
                        PReLU(),
                    )
                else:
                    raise ValueError(f"Unknown internal_model: {self.internal_model}")

                mlp.in_channels = input_dim
                layers.append(
                    GINEConv(
                        nn=mlp,
                        train_eps=True,
                        edge_dim=self.edge_input_dim,
                    )
                )
                input_dim = self.hidden_dim
            else:
                raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        return layers

    def _run_expert(
        self,
        expert_layers: ModuleList,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single expert stack with shared norm/activation and residual connections."""
        for i in range(self.num_layers):
            fx = x
            x = expert_layers[i](x, edge_index, edge_attr)
            x = self.norm_layers[i](x, batch_index)
            x = self.act_layers[i](x)
            x = x + fx if i > 0 else x
        return x

    def forward(self, graph: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch_index = (
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch,
        )

        # Compute gate weights from input node features
        gate_input = x
        gate_weights = self.gate_network(gate_input)  # [N_nodes, n_gated]

        # Run all experts
        expert_outputs = []
        for expert_layers in self.experts:
            expert_outputs.append(self._run_expert(expert_layers, x, edge_index, edge_attr, batch_index))

        if self.shared_expert:
            # Expert 0 is the shared (always-on) expert
            shared_output = expert_outputs[0]
            # Gate only experts 1..N
            gated_experts = torch.stack(expert_outputs[1:], dim=1)  # [N, n_gated, D]
            weighted = torch.sum(gated_experts * gate_weights.unsqueeze(-1), dim=1)  # [N, D]
            output = shared_output + weighted
        else:
            # Gate all experts
            all_experts = torch.stack(expert_outputs, dim=1)  # [N, n_experts, D]
            output = torch.sum(all_experts * gate_weights.unsqueeze(-1), dim=1)  # [N, D]

        return output
