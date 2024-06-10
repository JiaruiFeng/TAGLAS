from copy import deepcopy as c
from typing import (
    Optional,
    Callable, Any,
)

import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm

from TAGLAS.data import TAGDataset, TAGData, BaseDict
from TAGLAS.utils.dataset import generate_sample_split


class UltraChat200k(TAGDataset):
    r"""
    Convert UltraChat200k dataset to link graph format.
    """
    graph_description = "This is a chain graph constructed from an multi-round question answering conversation. Nodes represent a question or an answer and edges represent the order of the conversation. "

    def __init__(self,
                 name: str = "ultrachat200k",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, **kwargs)
        texts = {
            "x": self.side_data["node_texts"],
            "edge_attr": self.side_data["edge_texts"],
            "label": self.side_data["label_texts"],
            "question": self.side_data["question_texts"],
            "answer": self.side_data["answer_texts"]}
        self._data.update(texts)
        self.data = self._data.text_input_to_list()

    def raw_file_names(self) -> list:
        return []

    def download(self):
        pass

    def gen_data(self) -> tuple[list[TAGData], Any]:
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        edge_attr = ["Target sentence answer the instruction in the source sentence.",
                     "Target sentence is an instruction followed by the source answer."]
        all_text_list = []
        edge_index_list = []
        edge_map_list = []
        node_map_list = []
        question_list = []
        answer_list = []

        for i in tqdm(range(len(dataset))):
            message = dataset[i]["messages"]
            if len(message) % 2 != 0 or message[0]["role"] != "user" or len(message) <= 2:
                continue
            current_text_count = len(all_text_list)
            sample_text_list = [run["content"] for run in message]
            all_text_list.extend(sample_text_list)
            node_map = torch.arange(current_text_count, current_text_count + len(sample_text_list), dtype=torch.long)
            num_graphs = len(message) // 2 - 1
            for j in range(1, num_graphs + 1):
                row = torch.arange(j * 2 - 1)
                col = row + 1
                edge_index = torch.stack([row, col], dim=0).long()
                edge_map = torch.tensor([0] + [k for _ in range(j - 1) for k in range(1, -1, -1)], dtype=torch.long)
                edge_index_list.append(edge_index)
                edge_map_list.append(edge_map)
                node_map_list.append(node_map[0: j * 2])
                question_list.append(sample_text_list[j * 2])
                answer_list.append(sample_text_list[j * 2 + 1])

        unique_node_text, node_inverse_map = np.unique(np.array(all_text_list, dtype=object), return_inverse=True)
        unique_node_text = unique_node_text.tolist()
        node_inverse_map = torch.from_numpy(node_inverse_map).long()

        data_list = []
        for i, edge_index, edge_map, node_map in zip(range(len(node_map_list)), edge_index_list, edge_map_list,
                                                     node_map_list):
            graph = TAGData(
                edge_index=edge_index,
                node_map=node_inverse_map[node_map],
                edge_map=edge_map,
                question_map=torch.tensor([i], dtype=torch.long),
                answer_map=torch.tensor([i], dtype=torch.long),
                label_map=torch.tensor([i], dtype=torch.long),
            )
            data_list.append(graph)

        split_data = generate_sample_split(len(data_list))
        graph_split = BaseDict(train=split_data["train"][0],
                               val=split_data["val"][0],
                               test=split_data["test"][0])

        side_data = BaseDict(graph_split=graph_split,
                             question_texts=question_list,
                             node_texts=unique_node_text,
                             edge_texts=edge_attr,
                             label_texts=answer_list,
                             answer_texts=answer_list)
        return data_list, side_data

    def get_GQA_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the graph question answering tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        indexs = self.side_data.graph_split[split]
        label_map = self.label_map[indexs]
        labels = c(label_map)
        return indexs, labels, label_map.tolist()

    def get_GQA_list(self, label_map, **kwargs) -> tuple[list[tuple], np.ndarray, np.ndarray]:
        r"""Return question and answer list for graph question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        indexs = kwargs["indexs"]
        question_map = self.question_map[indexs]
        answer_map = self.answer_map[indexs]
        q_lists = self.question
        a_lists = self.answer

        question_list = []
        answer_list = []
        for q, a in zip(question_map, answer_map):
            question_list.append(q_lists[q])
            answer_list.append(a_lists[a])

        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        q_list, q_idxs = np.unique(np.array(question_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        q_list = q_list.tolist()
        label_map = [(q_idx, l_idx, a_idx) for q_idx, l_idx, a_idx in zip(q_idxs, label_map, a_idxs)]
        return label_map, q_list, a_list
