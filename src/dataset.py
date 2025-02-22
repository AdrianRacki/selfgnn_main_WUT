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
        """
        Returns the list of raw file names required for the dataset.

        This method defines the filenames of the input raw files that are
        necessary for the dataset. These files are expected to be located
        in the raw data directory.

        Returns:
            List[str]: A list containing the filenames of the raw input files.
        """
        return ["cations.csv", "anions.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Defines the filenames of the output processed files.

        This method returns a list of filenames that represent the processed
        graph dataset files. These files are typically used for storing
        preprocessed data that can be quickly loaded for training or inference
        purposes.

        Returns:
            List[str]: A list containing the filenames of the processed graph
            dataset files.
        """
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
            data = from_smiles(il_smiles)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

class LabeledGraphDataset(InMemoryDataset):
    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["filtered_database.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["labeled_graph_dataset.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        df = pd.read_csv(self.raw_paths[0])
        data_list = []
        for _, row in df.iterrows():
            il_smiles = row["smiles"]
            mp = row["MP"]
            data = from_smiles(il_smiles)
            data.y = mp
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
