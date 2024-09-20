from typing import (
    Union
)

import numpy as np
from torch import Tensor, LongTensor
from TAGLAS.utils.dataset import get_split_data
from TAGLAS.data import TAGDataset
from ..base import QATask
from ..process import subgraph_process


def default_text_labels(dataset: TAGDataset, split: str, **kwargs) -> tuple[LongTensor, Tensor, list, list, list]:
    r"""Obtain link-level question answering labels from dataset for the specified split.
    The dataset should implement get_LP_indexs_labels and get_LQA_list function.
    Args:
        dataset (TAGDataset): Dataset which implement the get_LP_indexs_labels and get_LQA_list function.
        split (str): Dataset split.
        kwargs: Other arguments.
    """
    sample_indexs, sample_labels, sample_label_maps = get_split_data(split, dataset.get_LP_indexs_labels)
    sample_label_maps, q_list, a_list = dataset.get_LQA_list(sample_label_maps, **kwargs)
    return sample_indexs, sample_labels, sample_label_maps, q_list, a_list


class LQATask(QATask):
    r"""Link-level question answering task.
    """

    def __process_split_and_label__(self):
        sample_indexs, sample_labels, sample_label_maps, q_list, a_list = default_text_labels(self.dataset, self.split)
        self.question_features = q_list
        self.answer_features = a_list
        return sample_indexs, sample_labels, sample_label_maps

    def __remove_link__(self, row: LongTensor, col: LongTensor, target_index: LongTensor):
        i, j = target_index
        remove_ind = np.logical_or(np.logical_and(row == i, col == j), np.logical_and(row == j, col == i), )
        keep_ind = np.logical_not(remove_ind).bool()
        return keep_ind

    def __process_graph__(
            self,
            index: Union[int, list, Tensor],
            edge_index: LongTensor,
            node_map: Tensor,
            edge_map: Tensor) -> tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
        edge_index, node_map, edge_map, target_index = subgraph_process(index, edge_index, node_map, edge_map,
                                                                        self.hop, self.max_nodes_per_hop,
                                                                        to_sparse=self.to_sparse, ppr_scores=self.ppr_scores)

        # remove the current training edge.
        keep_index = self.__remove_link__(edge_index[0], edge_index[1], target_index)
        edge_index = edge_index[:, keep_index]
        edge_map = edge_map[keep_index]
        return edge_index, node_map, edge_map, target_index

    def __dataset_prebuild__(self):
        # remove val and test edges from the dataset.
        edge_index = self.data["edge_index"][:, self.data["keep_edges"]]
        node_map = self.data["node_map"]
        edge_map = self.data["edge_map"]["keep_edges"]
        return edge_index, node_map, edge_map
