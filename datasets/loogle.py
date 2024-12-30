import json
import os.path as osp
from copy import deepcopy as c
from typing import (
    Optional,
    Callable, Any,
)

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import TAGDataset, TAGData, BaseDict
from TAGLAS.utils.io import extract_zip, download_hf_file


def construct_graph(sample):
    # title = sample["title"]
    # output = sample["output"]
    qa_pairs = eval(sample["qa_pairs"])
    graph = sample["graph"]
    node_txts = graph["entities"]
    edges = graph["edges"]
    edge_index = []
    edge_txts = []
    for edge in edges:
        edge_index.append([edge["source"], edge["target"]])
        edge_txts.append(edge["rel"])
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    return qa_pairs, node_txts, edge_txts, edge_index


class LooGLE(TAGDataset):
    r"""
    Scene graph dataset.
    """
    graph_description = "This is a graph generated from long article. "

    def __init__(self,
                 name: str = "loogle",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 subset: str = "longdep_qa",
                 **kwargs,
                 ) -> None:
        self.subset = subset
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
        return ["longdep_qa.json"]

    def download(self):
        download_hf_file(HF_REPO_ID, subfolder="loogle", filename="longdep_qa.json", local_dir=self.raw_dir)

    def gen_subset(self, subset):
        dataset = json.load(open(osp.join(self.raw_dir, f"{subset}.json")))
        node_txt_list = []
        edge_txt_list = []
        question_txt_list = []
        label_txt_list = []
        answer_txt_list = []
        graphs = []
        for obj in tqdm(dataset):
            qa_pairs, x, edge_attr, edge_index = construct_graph(obj)
            node_map = torch.arange(len(node_txt_list), len(node_txt_list) + len(x))
            node_txt_list.extend(x)
            edge_map = torch.arange(len(edge_txt_list), len(edge_txt_list) + len(edge_attr))
            edge_txt_list.extend(edge_attr)
            question_text = []
            answer_text = []
            supports_text = []
            for qa in qa_pairs:
                question_text.append(qa["Q"])
                answer_text.append(qa["A"])
                supports_text.append(qa["S"])
            label_text = answer_text
            question_map = torch.arange(len(question_txt_list), len(question_txt_list) + len(question_text))
            question_txt_list.extend(question_text)
            label_map = torch.arange(len(label_txt_list), len(label_txt_list) + len(label_text))
            label_txt_list.extend(label_text)
            answer_map = torch.arange(len(answer_txt_list), len(answer_txt_list) + len(answer_text))
            answer_txt_list.extend(answer_text)
            graphs.append((node_map, edge_map, edge_index, question_map, label_map, answer_map, supports_text))

        return node_txt_list, edge_txt_list, question_txt_list, label_txt_list, answer_txt_list, graphs

    def gen_data(self) -> tuple[list[TAGData], Any]:
        node_txt_list, edge_txt_list, question_txt_list, label_txt_list, answer_txt_list, graphs \
            = self.gen_subset(self.subset)
        unique_node_text, node_inverse_map = np.unique(np.array(node_txt_list, dtype=object), return_inverse=True)
        unique_edge_text, edge_inverse_map = np.unique(np.array(edge_txt_list, dtype=object), return_inverse=True)
        unique_question_text, question_inverse_map = np.unique(np.array(question_txt_list, dtype=object),
                                                               return_inverse=True)
        unique_label_text, label_inverse_map = np.unique(np.array(label_txt_list, dtype=object), return_inverse=True)
        unique_answer_text, answer_inverse_map = np.unique(np.array(answer_txt_list, dtype=object), return_inverse=True)

        unique_node_text = unique_node_text.tolist()
        unique_edge_text = unique_edge_text.tolist()
        node_inverse_map = torch.from_numpy(node_inverse_map).long()
        edge_inverse_map = torch.from_numpy(edge_inverse_map).long()
        unique_question_text = unique_question_text.tolist()
        unique_label_text = unique_label_text.tolist()
        unique_answer_text = unique_answer_text.tolist()
        question_inverse_map = torch.from_numpy(question_inverse_map).long()
        label_inverse_map = torch.from_numpy(label_inverse_map).long()
        answer_inverse_map = torch.from_numpy(answer_inverse_map).long()

        data_list = []
        id_list = []
        id = 0
        for node_map, edge_map, edge_index, question_map_list, label_map_list, answer_map_list, support_texts in graphs:
            for question_map, label_map, answer_map, supports_text in zip(question_map_list, label_map_list, answer_map_list, support_texts):
                data_list.append(
                    TAGData(node_map=node_inverse_map[node_map],
                            edge_index=edge_index,
                            edge_map=edge_inverse_map[edge_map],
                            label_map=label_inverse_map[label_map],
                            question_map=question_inverse_map[question_map],
                            answer_map=answer_inverse_map[answer_map],
                            supports_text=supports_text,
                            )
                )
                id_list.append(id)
                id += 1

        id_list = torch.tensor(id_list)
        train_idx = torch.arange(len(id_list))
        val_idx = torch.arange(len(id_list))
        test_idx = torch.arange(len(id_list))
        graph_split = BaseDict(train=train_idx, val=val_idx, test=test_idx)

        side_data = BaseDict(graph_split=graph_split,
                             question_texts=unique_question_text,
                             node_texts=unique_node_text,
                             edge_texts=unique_edge_text,
                             label_texts=unique_label_text,
                             answer_texts=unique_answer_text)

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

    def get_GQA_list(self, label_map: list, **kwargs) -> tuple[list[tuple], np.ndarray, np.ndarray]:
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
