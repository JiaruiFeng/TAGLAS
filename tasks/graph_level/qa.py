from copy import deepcopy as c
from typing import (
    Union,
    Any,
    Optional,
)

import torch
from torch import Tensor, LongTensor

from TAGLAS.data import TAGData, TAGDataset
from .prediction import DefaultTextGPTask
from ..process import value_to_tensor


def default_text_labels(dataset: TAGDataset, split: str, **kwargs) -> tuple[LongTensor, Tensor, list, list, list]:
    r"""Obtain graph-level question answering labels from dataset for the specified split.
    The dataset should implement get_GP_indexs_labels and get_GQA_list function.
    Args:
        dataset (TAGDataset): Dataset which implement the get_GP_indexs_labels and get_GQA_list function.
        split (str): Dataset split.
        kwargs: Other arguments.
    """

    sample_indexs, sample_labels, sample_label_maps = dataset.get_GQA_indexs_labels(split)
    sample_label_maps, q_list, a_list = dataset.get_GQA_list(sample_label_maps, indexs=sample_indexs, **kwargs)
    return sample_indexs, sample_labels, sample_label_maps, q_list, a_list


class GQATask(DefaultTextGPTask):
    r"""Graph-level question answering task.
    """

    def __process_split_and_label__(self):
        sample_indexs, sample_labels, sample_label_maps, q_list, a_list = default_text_labels(self.dataset, self.split)
        self.question_features = q_list
        self.answer_features = a_list
        return sample_indexs, sample_labels, sample_label_maps

    def __build_sample__(
            self,
            index: Union[int, Tensor, list],
            y: Union[int, float, Tensor,],
            label_map: Union[int, LongTensor, tuple],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor,
    ):
        index = value_to_tensor(index)
        edge_index, node_map, edge_map = self.__process_graph__(index, edge_index, node_map, edge_map)
        # for graph tasks, target index will be all nodes in the graph.
        new_index = torch.arange(len(node_map))
        question_map, label_map, answer_map = label_map
        label_map = value_to_tensor(label_map)
        question_map = value_to_tensor(question_map)
        answer_map = value_to_tensor(answer_map)
        y = value_to_tensor(y, to_long=False)

        return TAGData(edge_index=edge_index, node_map=node_map, y=y, label_map=label_map, target_index=new_index,
                       edge_map=edge_map,
                       question_map=question_map, answer_map=answer_map)

    def convert_text_to_embedding(
            self,
            encoder_name: str,
            encoder: Optional[Any] = None,
            convert_features: Optional[list[str]] = ["node", "edge", "label", "question", "answer"],
            from_saved: Optional[bool] = True) -> None:
        return self.__convert_text_to_embedding__(encoder_name, encoder, convert_features, from_saved)

    def __getitem__(self, item):
        data = c(self.data_list[item])
        node_map = data.node_map
        edge_map = data.edge_map
        label_map = data.label_map
        question_map = data.question_map
        answer_map = data.answer_map

        data.x = self.node_features[node_map]
        if self.edge_features is not None:
            data.edge_attr = self.edge_features[edge_map]
        data.label = self.label_features[label_map]
        data.question = self.question_features[question_map]
        data.answer = self.answer_features[answer_map]
        return data

    def collate(self, batch: list[TAGData], remap_keys: list[str] = ["node", "edge", "label", "question", "answer"]):
        return super(GQATask, self).collate(batch, remap_keys)
