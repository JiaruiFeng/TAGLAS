import os.path as osp
from abc import ABC, abstractmethod
from typing import (
    Optional,
    Callable,
    Any,
    Iterable,
)

import torch
from torch import Tensor
from torch_geometric.data import InMemoryDataset

from TAGLAS.constants import ROOT
from .data import TAGData


class TAGDataset(InMemoryDataset, ABC):
    r"""Base class for all TAG datasets. TAGDataset takes care of the dataset download, process, saving, loading,
    and split generation.
    Args:
        name (str): Name of the dataset.
        root (str, optional): Root directory of the dataset for saving and loading. Defaults to './TAGDataset'.
        transform (callable, optional): A Callable class to transform dataset after process.
        pre_transform (callable, optional): A Callable class to transform data before process.
        pre_filter (callable, optional): A Callable class to filter out sample in dataset.
        kwargs: Other arguments.
    """

    def __init__(
            self,
            name: str,
            root: Optional[str] = None,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            **kwargs) -> None:
        self.name = name
        root = (root if root is not None else ROOT)
        root = osp.join(root, self.name)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data = self._data.text_input_to_list()
        self.side_data = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self) -> list:
        return []

    @property
    def processed_file_names(self) -> list:
        r"""Names for all processed file that need to be saved.
        1. processed.pt: save a list of TAGData graphs.
        2. side_data.pt: Any side data that should be stored during preprocessing.
        """

        return ["processed.pkl", "side_data.pkl", ]


    @abstractmethod
    def gen_data(self) -> tuple[list[TAGData], Any]:
        r"""
        Subclass should implement this method, it should generate
        1, a list of TAG graphs
        2, any side data that should be stored during preprocessing
        """
        pass

    def process(self) -> None:
        data_list, side_data = self.gen_data()
        if side_data is not None:
            torch.save(side_data, self.processed_paths[1])
        else:
            torch.save("No side data", self.processed_paths[1])

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices,), self.processed_paths[0], pickle_protocol=4)

    def __str__(self):
        if "sub_name" in self.__dict__:
            return self.sub_name
        else:
            return self.name

    def get_sample(self, idx):
        data = self[idx]
        data.x = [self.x[i.squeeze()] for i in data.node_map]
        data.edge_attr = [self.edge_attr[i.squeeze()] for i in data.edge_map]
        data.label = [self.label[i.squeeze()] for i in data.label_map]
        return data

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        data_list = get_flattened_data_list([data for data in self])
        if isinstance(data_list[0].label_map, Tensor):
            label_map = torch.cat([data.label_map for data in data_list if 'label_map' in data], dim=0)
        else:
            label_map = torch.as_tensor([data.label_map for data in data_list if 'label_map' in data])

        if hasattr(self, '_data_list') and self._data_list is not None:
            self._data_list = self.len() * [None]
        return self._infer_num_classes(label_map)


def get_flattened_data_list(data_list: Iterable[Any]) -> list[TAGData]:
    outs: list[TAGData] = []
    for data in data_list:
        if isinstance(data, TAGData):
            outs.append(data)
        elif isinstance(data, (tuple, list)):
            outs.extend(get_flattened_data_list(data))
        elif isinstance(data, dict):
            outs.extend(get_flattened_data_list(data.values()))
    return outs
