import os.path as osp
from typing import (
    Optional,
    Callable,
    Any
)

import numpy as np
import torch
from numpy.random import randint
from torch import Tensor
from torch import LongTensor
from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import TAGData, TAGDataset, BaseDict
from TAGLAS.utils.io import download_hf_file
from .gen_data import gen_graph, get_raw_dataset, NAME_TO_SPLIT


class Chembl(TAGDataset):
    r"""Chembl molecules instruction dataset collection. Get available datasets by Chembl.available_datasets
    """
    available_datasets = list(NAME_TO_SPLIT.keys())
    graph_description = "This graph is a molecule. Nodes represent chemical atoms and edge represent chemical bonds. "

    def __init__(self,
                 name: str = "pcba",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs,
                 ) -> None:
        self.sub_name = name
        super().__init__("chembl", root, transform, pre_transform, pre_filter, **kwargs)
        answer = self.side_data["label_texts"]
        label = self.__process_label_features__(answer)
        texts = {
            "x": self.side_data["node_texts"],
            "edge_attr": self.side_data["edge_texts"],
            "answer": answer,
            "question": self.side_data["question_texts"],
            "label": label,
        }
        self._data.update(texts)
        self._data = self._data.text_input_to_list()

    @property
    def raw_file_names(self) -> list:
        return ["id2element.csv", "mol_label_desc.json", "prompt_pretrain.json"]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.sub_name, 'processed')

    def download(self):
        download_hf_file(HF_REPO_ID, subfolder="chemmol", filename="id2element.csv", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="chemmol", filename="mol_label_desc.json", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="chemmol", filename="prompt_pretrain.json", local_dir=self.raw_dir)

    def gen_data(self) -> tuple[list[TAGData], Any]:
        graphs, label_texts, question_texts = get_raw_dataset(self.sub_name, self.raw_dir)
        pyg_graph, graph_texts, split = gen_graph(graphs, self.sub_name)
        side_data = BaseDict(graph_split=split,
                             question_texts=question_texts,
                             node_texts=graph_texts[0],
                             edge_texts=graph_texts[1],
                             label_texts=label_texts)

        return [d for d in pyg_graph], side_data

    def __process_label_features__(self, answer_features: list[str]):
        labels = []
        answers = answer_features
        if self.sub_name in ["esol", "freesolv", "lipo"]:
            prefix_dict = {
                "esol": 165,
                "freesolv": 288,
                "lipo": 308,
            }
            for i in range(len(answers)):
                labels.append(answers[i][prefix_dict[self.sub_name]:])
        elif self.sub_name == "molproperties":
            labels = answer_features
        else:
            for i in range(len(answers)):
                labels.append(answers[i][:-1].split("is")[-1].lower().strip())
        return labels

    def get_GP_indexs_labels(self, split: str = "train") -> tuple[LongTensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the graph-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """

        indexs = self.side_data.graph_split[split]
        label_map = self.label_map[indexs]

        prefix_dict = {
            "esol": 165,
            "freesolv": 288,
            "lipo": 308,
        }
        if self.sub_name in ["esol", "freesolv", "lipo"]:
            label = np.array(self.answer)
            labels = torch.tensor([float(v[prefix_dict[self.sub_name]:]) for v in label[label_map]])
        elif self.sub_name == "molproperties":
            labels = self.label_map
        else:
            cum_label_map = self.cum_label_map[indexs]
            labels = label_map
            label_map = cum_label_map

        return torch.tensor(indexs, dtype=torch.long), labels.squeeze(), label_map.tolist()

    def get_GQA_indexs_labels(self, split: str = "train") -> tuple[LongTensor, Tensor, list[list]]:
        r"""Return sample labels and their corresponding index for the graph question answering tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """

        indexs, labels, label_map = self.get_GP_indexs_labels(split)
        if len(labels.size()) == 1:
            labels = labels.unsqueeze(-1)
        if self.sub_name in ["esol", "freesolv", "lipo"]:
            return indexs, labels, label_map
        elif self.sub_name == "molproperties":
            question_map = self.question_map[indexs]
            new_indexs = []
            new_label_map = []
            new_labels = []
            num_sample = len(label_map)
            num_task = len(label_map[0])
            for i in range(num_sample):
                for j in range(num_task):
                    if label_map[i][j] == -1:  # label not available
                        continue
                    new_indexs.append(indexs[i])
                    new_label_map.append([question_map[i][j], label_map[i][j]])
                    new_labels.append(labels[i][j])
        else:
            new_indexs = []
            new_label_map = []
            new_labels = []
            num_sample = len(label_map)
            num_task = len(label_map[0])
            for i in range(num_sample):
                for j in range(num_task):
                    if label_map[i][j] == -1:  # label not available
                        continue
                    new_indexs.append(indexs[i])
                    new_label_map.append([j, label_map[i][j]])
                    new_labels.append(labels[i][j])

        return torch.tensor(new_indexs).long(), torch.tensor(new_labels).long(), new_label_map

    def get_GQA_list(self, label_map: list, **kwargs) -> tuple[list[tuple], list, list]:
        r"""Return question and answer list for graph question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """

        num_sample = len(label_map)
        q_lists = self.question
        label_features = self.answer
        answer_list = []

        if self.sub_name in ["esol", "freesolv", "lipo"]:
            prefix_dict = {
                "esol": 165,
                "freesolv": 288,
                "lipo": 308,
            }

            q_idxs = randint(len(q_lists[0]), size=num_sample).tolist()
            for l in label_map:
                answer_list.append(label_features[l][prefix_dict[self.sub_name]:] + ".")
            a_list, a_idxs = np.unique(answer_list, return_inverse=True)
            a_list = a_list.tolist()
            q_list = q_lists[0]

            label_map = [(q_idx, l_idx, a_idx) for q_idx, l_idx, a_idx in zip(q_idxs, label_map, a_idxs)]
            return label_map, q_list, a_list

        elif self.sub_name == "molproperties":
            question_list = []
            for q_idx, j in label_map:
                label_feature = label_features[j]
                answer = label_feature + "."
                answer_list.append(answer)
                question_list.append(q_lists[q_idx])

            a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
            q_list, q_idxs = np.unique(np.array(question_list, dtype=object), return_inverse=True)
            a_list = a_list.tolist()
            q_list = q_list.tolist()
            label_map = [(q_idx, l_idx[1], a_idx) for q_idx, l_idx, a_idx in zip(q_idxs, label_map, a_idxs)]
            return label_map, q_list, a_list

        else:
            question_list = []
            for i, j in label_map:
                q_idx = randint(len(q_lists[i]), size=1)[0]
                label_feature = label_features[j]
                answer = label_feature[:-1].split("is")[-1].lower().strip() + "."
                answer_list.append(answer)
                question_list.append(q_lists[i][q_idx])

            a_list, a_idxs = np.unique(answer_list, return_inverse=True)
            q_list, q_idxs = np.unique(question_list, return_inverse=True)
            a_list = a_list.tolist()
            q_list = q_list.tolist()
            label_map = [(q_idx, l_idx[1], a_idx) for q_idx, l_idx, a_idx in zip(q_idxs, label_map, a_idxs)]
            return label_map, q_list, a_list
