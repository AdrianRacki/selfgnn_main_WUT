from typing import List

import pandas as pd
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from utils import from_smiles, add_global_features
import torch

class LabeledGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        node_features,
        edge_features,
        raw_filename: str,
        processed_filename: str,
        global_features,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.node_features = node_features.features
        self.edge_features = edge_features.features
        self.global_features = global_features.features
        self.raw_filename = [raw_filename]
        self.processed_filename = [processed_filename]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return self.raw_filename

    @property
    def processed_file_names(self) -> List[str]:
        return self.processed_filename

    def download(self) -> None:
        pass

    def process(self) -> None:
        df = pd.read_csv(self.raw_paths[0])
        data_list = []
        for _, row in tqdm(df.iterrows(), desc="Processing data", total=len(df)):
            il_smiles = row["smiles"]
            mp = row["value"]
            data = from_smiles(
                smiles=il_smiles,
                node_features=self.node_features,
                edge_features=self.edge_features,
            )
            data.y = float(mp)
            data = add_global_features(self.global_features, data)
            if data.x.shape[0] == 0 or data.edge_index.shape[0] == 0:  # type: ignore
                continue
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])



