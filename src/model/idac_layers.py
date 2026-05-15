import torch
from torch import nn
from torch_geometric.data import Data


class IDACProjector(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        dropout_rate: float = 0.15,
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
        temperature_size = 1
        self.temperature_projector = torch.nn.Identity()
        in_features = in_features + temperature_size + global_features_size if append_global_features else in_features + temperature_size
        self.input_batchnorm = torch.nn.BatchNorm1d(in_features)
        sizes = [in_features, in_features // 2, in_features // 2, in_features // 4, out_features]

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


class IDACTransformerProjector(nn.Module):
    """Token-based transformer projector for IDAC prediction.

    Treats per-molecule pooled embeddings, temperature, and a learnable
    interaction token as a sequence, runs self-attention, and uses the
    interaction token for final prediction.

    Token order: [mol_0, mol_1, …, mol_{n-1}, temperature, interaction]
    """

    def __init__(
        self,
        in_features: int,
        num_mols: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 128,
        dropout_rate: float = 0.1,
        out_features: int = 1,
        append_global_features: bool = True,
        global_features_size: int = 45,
    ):
        super().__init__()
        self.num_mols = num_mols
        self.out_features = out_features
        self.append_global_features = append_global_features
        self.per_mol_dim = in_features // num_mols

        global_feat_per_mol = global_features_size // num_mols if append_global_features else 0
        mol_input_dim = self.per_mol_dim + global_feat_per_mol

        # Project each mol token to d_model (separate per molecule role)
        self.mol_projectors = nn.ModuleList([nn.Linear(mol_input_dim, d_model) for _ in range(num_mols)])

        # Temperature: scalar → d_model via small MLP
        self.temp_projector = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 4, d_model),
        )

        # Learnable interaction (CLS) token
        self.interaction_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.interaction_token, std=0.02)

        # Learnable positional embeddings for all tokens
        num_tokens = num_mols + 2  # mol tokens + temperature + interaction
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Normalize assembled tokens before transformer
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder (Pre-LN for stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, out_features),
        )

    def forward(self, graph: Data) -> torch.Tensor:
        B = graph.x.shape[0]

        # 1. Per-molecule tokens
        if graph.x.dim() == 3:
            mol_tokens = graph.x  # [B, num_mols, per_mol_dim]
        else:
            mol_tokens = graph.x.view(B, self.num_mols, self.per_mol_dim)

        # 2. Optionally append global features per mol
        if self.append_global_features:
            g = graph.g.view(B, self.num_mols, -1)  # [B*num_mols, feats] → [B, num_mols, feats]
            mol_tokens = torch.cat([mol_tokens, g], dim=-1)

        mol_tokens = torch.stack([self.mol_projectors[i](mol_tokens[:, i]) for i in range(self.num_mols)], dim=1)  # [B, num_mols, d_model]

        # 3. Temperature token
        temp = graph.temperature.float().view(-1, 1)
        temp_token = self.temp_projector(temp).unsqueeze(1)  # [B, 1, d_model]

        # 4. Interaction (CLS) token
        cls_token = self.interaction_token.expand(B, -1, -1)  # [B, 1, d_model]

        # 5. Assemble sequence: [mol_0, …, mol_{n-1}, temp, interaction]
        tokens = torch.cat([mol_tokens, temp_token, cls_token], dim=1)  # [B, num_mols+2, d_model]
        tokens = self.input_norm(tokens + self.pos_embedding)

        # 6. Transformer self-attention
        tokens = self.transformer_encoder(tokens)

        # 7. Extract interaction token (last position) → output
        interaction = self.output_norm(tokens[:, -1, :])
        return self.output_mlp(interaction)
