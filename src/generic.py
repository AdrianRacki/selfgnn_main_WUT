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


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 2,
        use_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """
        Multi-Layer Perceptron with customizable architecture.
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() if use_norm else None
        self.dropout = torch.nn.ModuleList()

        sizes = []
        if num_layers == 1:
            sizes = [out_features]
        else:
            for i in range(num_layers):
                size = in_features - i * (in_features - out_features) // (
                    num_layers - 1
                )
                sizes.append(size)

        current_size = in_features
        for i, size in enumerate(sizes):
            self.layers.append(torch.nn.Linear(current_size, size))
            current_size = size

            if i < num_layers - 1:
                self.activations.append(torch.nn.PReLU())
                if dropout_rate > 0:
                    self.dropout.append(torch.nn.Dropout(dropout_rate))
                if use_norm:
                    self.norms.append(torch.nn.BatchNorm1d(size))  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
                if len(self.dropout) > 0:
                    x = self.dropout[i](x)
                if self.norms is not None:
                    x = self.norms[i](x)
        return x

class Projector(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        dropout_rate: float = 0.3,
    ):
        """
        Projector layer with customizable architecture.
        Returns both final output and 16-dim embedding.
        """
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        self.out_features = out_features
        
        sizes = [in_features, 64, 32, 16, 1]
        
        for i in range(len(sizes) - 1):
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:  
                self.activations.append(torch.nn.PReLU())
                self.norms.append(torch.nn.BatchNorm1d(sizes[i+1]))
                if dropout_rate > 0:
                    self.dropout.append(torch.nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
                x = self.norms[i](x)
                if len(self.dropout) > 0:
                    x = self.dropout[i](x)
            if i == 2:  # This corresponds to the 16-dim output
                mol_embedding = x
        return x, mol_embedding # type: ignore
