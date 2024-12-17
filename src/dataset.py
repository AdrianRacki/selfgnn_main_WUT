import itertools
from typing import List

import pandas as pd
from torch_geometric.data import InMemoryDataset

from graph_tools import from_smiles


class SelfGraphDataset(InMemoryDataset):
    """Defines torch_geometric.data.InMemoryDataset implementation to
    create dataset of all combinations of ions in from to .csv files.

    Args:
        InMemoryDataset (torch_geometric.data.InMemoryDataset): Base class
        for creating PyG InMemoryDatasets.
    """

    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Defines filename of input raw files."""
        return ["cations.csv", "anions.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        """Defines filename of output processed files."""
        return ["graph_dataset.pt"]

    def download(self) -> None:
        """Not implemented - files are stored locally."""
        pass

    def process(self) -> None:
        """Takes data from raw directory and generate list of Data (graph) to save as processed.
        Possible filters and pre-transforms on each element of data list."""
        cations_df = pd.read_csv(self.raw_paths[0])
        anions_df = pd.read_csv(self.raw_paths[1])
        data_list = []
        for il in itertools.product(cations_df["cation"], anions_df["anion"]):
            il_smiles = ".".join(il)
            data_list.append(from_smiles(il_smiles))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
