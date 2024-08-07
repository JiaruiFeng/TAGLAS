from copy import deepcopy as c
import types
from typing import (
    Union,
    Any,
    Optional,
)

import torch
from torch import Tensor, LongTensor
from TAGLAS.utils.dataset import get_split_data
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

    sample_indexs, sample_labels, sample_label_maps = get_split_data(split, dataset.get_GQA_indexs_labels)
    sample_label_maps, q_list, a_list = dataset.get_GQA_list(sample_label_maps, indexs=sample_indexs, **kwargs)
    return sample_indexs, sample_labels, sample_label_maps, q_list, a_list


class GQATask(DefaultTextGPTask):
    r"""Graph-level question answering task.
    """

    def __sampling__(self, num_samples: int, num_selected_samples: int) -> list:
        sample_label_map = self.sample_label_map
        sample_mode = self.sample_mode
        sample_label_map = [label_map[1] for label_map in sample_label_map]
        if sample_mode != "random":
            # handle graph data which label is list of list
            if isinstance(sample_label_map[1], list):
                if len(sample_label_map[0]) == 1:
                    sample_label_map = [lbs[0] for lbs in sample_label_map]
                else:
                    print(f'Contains multiple labels per sample, use randomly sampling instead.')
                    return self.__random_sampling__(num_samples, num_selected_samples)

            label_map_set = set(sample_label_map)
            num_unique_label = len(label_map_set)
            if num_unique_label > num_samples / 2:
                print(f'Probably not the classification task, use randomly sampling instead.')
                return self.__random_sampling__(num_samples, num_selected_samples)

            if sample_mode == "balanced":
                return self.__balanced_sampling__(sample_label_map, num_unique_label, num_selected_samples)
            elif sample_mode == "stratified":
                return self.__stratified_sampling__(sample_label_map, num_samples, num_selected_samples)
            else:
                raise ValueError(f"sample mode {sample_mode} is not supported. Please choose from random, balanced, or stratified.")

        return self.__random_sampling__(num_samples, num_selected_samples)

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
            convert_features: list[str] = ["node", "edge", "label", "question", "answer"],
            from_saved: bool = True) -> None:
        return self.__convert_text_to_embedding__(encoder_name, encoder, convert_features, from_saved)

    def __getitem__(self, item):
        data = c(self.data_list[item])
        node_map = data.node_map
        edge_map = data.edge_map
        label_map = data.label_map
        question_map = data.question_map
        answer_map = data.answer_map

        data.x = self.node_features[node_map.numpy()]
        if self.edge_features is not None:
            data.edge_attr = self.edge_features[edge_map.numpy()]
        data.label = self.label_features[label_map.numpy()]
        data.question = self.question_features[question_map.numpy()]
        data.answer = self.answer_features[answer_map.numpy()]
        if self.post_funcs is not None:
            if isinstance(self.post_funcs, types.FunctionType):
                post_funcs = [self.post_funcs]
            else:
                assert isinstance(self.post_funcs, list)
                post_funcs = self.post_funcs
            for post_func in post_funcs:
                data = post_func(data, task_class=self)
        return data

    def collate(self, batch: list[TAGData], remap_keys: list[str] = ["node", "edge", "label", "question", "answer"]):
        return super(GQATask, self).collate(batch, remap_keys)
