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

from TAGLAS.data import TAGDataset, TAGData, BaseDict
import datasets

class WebQSP(TAGDataset):
    r"""
    WebQSP dataset.
    """
    graph_description = "This is a knowledge graph generated from Wiki data. Nodes represent entities and edges represent the relationship between two entities. "

    def __init__(self,
                 name: str = "webqsp",
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
        return ["train_sceneGraphs.json", "questions.csv", "scene_graph_split.pt"]


    def gen_data(self) -> tuple[list[TAGData], Any]:
        raw_datasets = datasets.load_dataset('rmanluo/RoG-webqsp')
        node_txt_list = []
        edge_txt_list = []
        question_txt_list = []
        answer_txt_list = []
        label_txt_list = []
        graph_list = []
        count = 0
        for dataset in [raw_datasets["train"], raw_datasets["validation"], raw_datasets["test"]]:
            for example in tqdm(dataset):
                question_txt_list.append(example["question"])
                answer = ('|').join(example['answer']).lower()
                answer_txt_list.append(answer)
                label_txt_list.append(answer)
                raw_nodes = {}
                raw_edges = []
                for tri in example["graph"]:
                    h, r, t = tri
                    h = h.lower()
                    t = t.lower()
                    if h not in raw_nodes:
                        raw_nodes[h] = len(raw_nodes)
                    if t not in raw_nodes:
                        raw_nodes[t] = len(raw_nodes)
                    raw_edges.append({
                        "src": raw_nodes[h],
                        "edge_attr": r,
                        "dst": raw_nodes[t]
                    })
                nodes = pd.DataFrame([{
                    "node_id": v,
                    "node_attr": k,
                } for k, v in raw_nodes.items()],
                    columns=["node_id", "node_attr"])
                edges = pd.DataFrame(raw_edges,
                                     columns=["src", "edge_attr", "dst"])
                nodes.node_attr = nodes.node_attr.fillna("")
                x = nodes.node_attr.tolist()
                node_map = torch.arange(len(node_txt_list), len(node_txt_list) + len(x))
                edge_attr = edges.edge_attr.tolist()
                edge_map = torch.arange(len(edge_txt_list), len(edge_txt_list) + len(edge_attr))
                node_txt_list.extend(x)
                edge_txt_list.extend(edge_attr)

                label_map = torch.tensor([count]).long()
                graph_list.append((nodes, edges, node_map, edge_map, label_map, label_map, label_map))
                count += 1


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
        for nodes, edges, node_map, edge_map, label_map, answer_map, question_map in graph_list:
            edge_index = torch.tensor([
                edges.src.tolist(),
                edges.dst.tolist(),
            ], dtype=torch.long)

            data_list.append(
                TAGData(node_map=node_inverse_map[node_map],
                        edge_index=edge_index,
                        edge_map=edge_inverse_map[edge_map],
                        label_map=label_inverse_map[label_map],
                        question_map=question_inverse_map[question_map],
                        answer_map=answer_inverse_map[answer_map]
                        )
            )
        train_idx = torch.arange(len(raw_datasets["train"]), dtype=torch.long)
        val_idx = torch.arange(len(raw_datasets["validation"]), dtype=torch.long) + len(raw_datasets["train"])
        test_idx = (torch.arange(len(raw_datasets["test"]), dtype=torch.long) + len(raw_datasets["train"])
                    + len(raw_datasets["validation"]))
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
