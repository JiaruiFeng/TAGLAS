import re
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
from TAGLAS.utils.io import download_hf_file


class ExplaGraph(TAGDataset):
    graph_description = "This is a graph constructed from commonsense logic. Nodes represent commonsense objects and edges represent the relation between two objects."

    def __init__(self,
                 name: str = "expla_graph",
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
            "question": self.side_data["question_texts"]}
        self._data.update(texts)
        self.data = self._data.text_input_to_list()

    def raw_file_names(self) -> list:
        return ["train_dev.tsv", "expla_graph_split.pt"]

    def download(self):
        download_hf_file(HF_REPO_ID, subfolder="explagraph", filename="expla_graph_split.pt", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="explagraph", filename="train_dev.tsv", local_dir=self.raw_dir)

    def textualize_graph(self, graph: str):
        triplets = re.findall(r'\((.*?)\)', graph)
        nodes = {}
        edges = []
        for tri in triplets:
            src, edeg_attr, dst = tri.split(';')
            src = "Common sense concept: " + src.lower().strip()
            dst = "Common sense concept: " + dst.lower().strip()
            if src not in nodes:
                nodes[src] = len(nodes)
            if dst not in nodes:
                nodes[dst] = len(nodes)
            edges.append({'src': nodes[src],
                          'edge_attr': "Common sense relation: " + edeg_attr.lower().strip(),
                          'dst': nodes[dst], })

        nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
        edges = pd.DataFrame(edges)
        return nodes, edges

    def gen_data(self) -> tuple[list[TAGData], Any]:
        prompt = 'Do argument 1 and argument 2 support or counter each other? Please only answer support or counter.'
        dataset = pd.read_csv(self.raw_paths[0], sep='\t')
        node_txt_list = []
        edge_txt_list = []
        question_txt_list = []
        label_txt_list = []
        graphs = []

        for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
            nodes, edges = self.textualize_graph(row['graph'])
            question_txt_list.append(f'Argument 1: {row.arg1}\nArgument 2: {row.arg2}\n{prompt}')
            question_map = torch.tensor([i], dtype=torch.long)
            label_txt_list.append(row.label)
            label_map = torch.tensor([i], dtype=torch.long)
            edge_index = torch.LongTensor([edges.src, edges.dst])
            x = nodes.node_attr.tolist()
            edge_attr = edges.edge_attr.tolist()
            node_map = torch.arange(len(node_txt_list), len(node_txt_list) + len(x))
            node_txt_list.extend(x)
            edge_map = torch.arange(len(edge_txt_list), len(edge_txt_list) + len(edge_attr))
            edge_txt_list.extend(edge_attr)
            graphs.append((node_map, edge_map, edge_index, question_map, label_map))

        unique_node_text, node_inverse_map = np.unique(np.array(node_txt_list, dtype=object), return_inverse=True)
        unique_edge_text, edge_inverse_map = np.unique(np.array(edge_txt_list, dtype=object), return_inverse=True)
        unique_question_text, question_inverse_map = np.unique(np.array(question_txt_list, dtype=object),
                                                               return_inverse=True)
        unique_label_text, label_inverse_map = np.unique(np.array(label_txt_list, dtype=object), return_inverse=True)

        unique_node_text = unique_node_text.tolist()
        unique_edge_text = unique_edge_text.tolist()
        node_inverse_map = torch.from_numpy(node_inverse_map).long()
        edge_inverse_map = torch.from_numpy(edge_inverse_map).long()
        unique_question_text = unique_question_text.tolist()
        unique_label_text = unique_label_text.tolist()
        question_inverse_map = torch.from_numpy(question_inverse_map).long()
        label_inverse_map = torch.from_numpy(label_inverse_map).long()

        data_list = []
        for node_map, edge_map, edge_index, question_map, label_map in graphs:
            data_list.append(
                TAGData(node_map=node_inverse_map[node_map],
                        edge_index=edge_index,
                        edge_map=edge_inverse_map[edge_map],
                        label_map=label_inverse_map[label_map],
                        question_map=question_inverse_map[question_map],
                        )
            )

        graph_split = BaseDict(**torch.load(self.raw_paths[1]))
        side_data = BaseDict(graph_split=graph_split,
                             question_texts=unique_question_text,
                             node_texts=unique_node_text,
                             edge_texts=unique_edge_text,
                             label_texts=unique_label_text)

        return data_list, side_data

    def get_GP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the graph-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        indexs = self.side_data.graph_split[split]
        label_map = self.label_map[indexs]
        labels = c(label_map)
        return indexs, labels, label_map.tolist()


    def get_GQA_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the graph question answering tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        return self.get_GP_indexs_labels(split)

    def get_GQA_list(self, label_map: list, **kwargs) -> tuple[list[tuple], np.ndarray, np.ndarray]:
        r"""Return question and answer list for graph question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        indexs = kwargs["indexs"]
        q_lists = self.side_data.question_texts
        label_features = self.label
        question_map = self.question_map[indexs].tolist()

        question_list = []
        answer_list = []
        for l, q in zip(label_map, question_map):
            question_list.append(q_lists[q])
            answer_list.append(label_features[l] + ".")

        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        q_list, q_idxs = np.unique(np.array(question_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        q_list = q_list.tolist()

        label_map = [(q_idx, l_idx, a_idx) for q_idx, l_idx, a_idx in zip(q_idxs, label_map, a_idxs)]
        return label_map, q_list, a_list
