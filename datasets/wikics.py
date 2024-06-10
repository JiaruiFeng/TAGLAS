import functools
import json
from itertools import chain
from typing import (
    Optional,
    Callable,
    Any,
)

import numpy as np
import torch
from torch import Tensor

from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import BaseDict
from TAGLAS.data.dataset import TAGDataset, TAGData
from TAGLAS.utils.graph import safe_to_undirected
from TAGLAS.utils.io import download_hf_file


class WikiCS(TAGDataset):
    """Wikipedia link network. Contains 10 classes for node classification.
    """
    graph_description = "This is a Wikipedia graph focusing on computer science. Nodes represent Wikipedia terms and edges represent two terms that have hyperlinks. "

    def __init__(self,
                 name: str = "wikics",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, **kwargs)

    def raw_file_names(self) -> list:
        return ["data.json", "metadata.json", "wikics.json"]

    def download(self) -> None:
        download_hf_file(HF_REPO_ID, subfolder="wikics", filename="metadata.json", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="wikics", filename="data.json", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="wikics", filename="wikics.json", local_dir=self.raw_dir)

    def gen_data(self) -> tuple[list[TAGData], Any]:
        with open(self.raw_paths[0], 'r') as f:
            data = json.load(f)

        x = torch.tensor(data['features'], dtype=torch.float)
        y = torch.tensor(data['labels'], dtype=torch.long)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))  # type: ignore
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index, _ = safe_to_undirected(edge_index)

        train_mask = torch.tensor(data['train_masks'], dtype=torch.bool)
        train_mask = train_mask.t().contiguous()

        val_mask = torch.tensor(data['val_masks'], dtype=torch.bool)
        val_mask = val_mask.t().contiguous()

        test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)

        stopping_mask = torch.tensor(data['stopping_masks'], dtype=torch.bool)
        stopping_mask = stopping_mask.t().contiguous()

        with open(self.raw_paths[1], 'r') as f:
            raw_data = json.load(f)

        node_info = raw_data["nodes"]
        label_info = raw_data["labels"]
        node_text_lst = []
        label_text_lst = []

        # Process Node text
        for node in node_info:
            node_feature = (
                ("Wikipedia entry. Entry name: " + node["title"] + ". Entry content: " +
                 functools.reduce(lambda x, y: x + " " + y, node["tokens"])).strip())
            node_text_lst.append(node_feature)

        node_map = torch.arange(len(node_text_lst), dtype=torch.long)

        # Process Label text
        for label in label_info.values():
            label_feature = label.lower().strip()
            label_text_lst.append(label_feature)

        # define single edge text
        edge_text_lst = ["Page link between two Wikipedia entries."]
        edge_map = torch.zeros(edge_index.size(-1), dtype=torch.long)

        # additional label description.
        with open(self.raw_paths[-1]) as f:
            label_desc = json.load(f)

        ordered_desc = BaseDict()
        for i in range(len(label_desc)):
            label = label_desc[i]["name"].lower()
            desc = label_desc[i]["description"]
            ordered_desc[label] = desc

        data = TAGData(x=node_text_lst,
                       node_map=node_map,
                       edge_index=edge_index,
                       edge_attr=edge_text_lst,
                       edge_map=edge_map,
                       x_original=x,
                       label=label_text_lst,
                       label_map=y, )

        side_data = BaseDict(node_split=
                             BaseDict(train=train_mask,
                                      val=val_mask,
                                      test=test_mask,
                                      stopping=stopping_mask),
                             label_description=ordered_desc)

        return [data], side_data

    def get_NP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the node-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        mask = self.side_data.node_split[split]
        if split != "test":
            mask = mask[:, 0]
        indexs = torch.where(mask)[0]
        labels = self.label_map[indexs]
        label_map = labels
        return indexs, labels, label_map.tolist()

    def get_NQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for node question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["What is the most likely category for target Wikipedia term?"]
        answer_list = []
        label_features = self.label
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]
        return label_map, q_list, a_list
