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
from copy import deepcopy as c
import numpy as np


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

    def _map_to_feature(self, features, feature_map):
        if len(feature_map.shape) == 1:
            return [features[i] for i in feature_map]
        elif len(feature_map.shape) == 0:
            raise ValueError("Map should be at least 1-d array")
        else:
            return [self._map_to_feature(features, feature_map[..., i]) for i in range(feature_map.shape[-1])]

    def __getitem__(self, item):
        data = super(TAGDataset, self).__getitem__(item)
        missed_key = []
        for key in ["x", "edge_attr", "label"]:
            if key not in data:
                missed_key.append(key)
        if len(missed_key) == 0:
            return data

        data = c(data)
        update_dict = {}
        for key in missed_key:
            if key not in self._data:
                continue
            features = getattr(self, key)

            if key == "x":
                x = self._map_to_feature(features, data.node_map.numpy())
                update_dict["x"] = x
            elif key == "edge_attr":
                edge_attr = self._map_to_feature(features, data.edge_map.numpy())
                update_dict["edge_attr"] = edge_attr
            else:
                label = self._map_to_feature(features, data.label_map.numpy())
                update_dict["label"] = label

        data.update(update_dict)
        return data


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
