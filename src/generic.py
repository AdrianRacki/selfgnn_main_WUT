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
