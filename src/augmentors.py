from abc import ABC, abstractmethod
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, mask_feature


class Augmentor(ABC):
    """Base class for graph augmentors. DataBatch is also supported."""

    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Data) -> Data:
        pass

    def __call__(self, g: Data) -> Data:
        return self.augment(g)


class Compose(Augmentor):
    """
    Composes several augmentations together.

    Args:
        augmentors (list[Augmentor]): List of augmentors to compose.
    """

    def __init__(self, augmentors: list[Augmentor]) -> None:
        super().__init__()
        self.augmentors = augmentors

    def augment(self, g: Data) -> Data:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class EdgeRemoving(Augmentor):
    """
    Removes edges from the graph with a given probability.

    Args:
        p (float): Probability of removing an edge.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def augment(self, g: Data) -> Data:
        g_aug = deepcopy(g)
        g_aug.edge_index, g_aug.edge_attr = dropout_adj(
            edge_index=g.edge_index, # type: ignore
            edge_attr=g.edge_attr,
            p=self.p,
            force_undirected=True,
        )  
        return g_aug


class FeatureMasking(Augmentor):
    """
    Masks features of the graph with a given probability.

    Args:
        p (float): Probability of masking a feature.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def augment(self, g: Data) -> Data:
        g_aug = deepcopy(g)
        g_aug.x, _ = mask_feature(g.x, self.p, mode="all")  # type: ignore
        return g_aug
