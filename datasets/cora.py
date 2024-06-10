import json
from typing import (
    Optional,
    Callable,
    Any
)

import numpy as np
import torch
from torch import Tensor

from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import TAGData, TAGDataset, BaseDict
from TAGLAS.utils.dataset import generate_link_split
from TAGLAS.utils.graph import safe_to_undirected
from TAGLAS.utils.io import download_hf_file


class Cora(TAGDataset):
    """Cora co-citation network dataset.
    """
    graph_description = "This is a co-citation network focusing on artificial intelligence, nodes represent academic papers and edges represent two papers that are co-cited by other papers. "

    def __init__(self,
                 name: str = "cora",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, **kwargs)
        # Generate random split for link prediction.
        self.side_data.link_split, self.side_data.keep_edges = generate_link_split(self._data.edge_index)

    def raw_file_names(self) -> list:
        return ["cora.pt", "cora_node.json"]

    def download(self) -> None:
        download_hf_file(HF_REPO_ID, subfolder="Cora", filename="cora.pt", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="Cora", filename="cora_node.json", local_dir=self.raw_dir)

    def gen_data(self) -> tuple[list[TAGData], Any]:
        cora_data = torch.load(self.raw_paths[0])
        data = TAGData(**cora_data.to_dict())
        delattr(data, "raw_text")

        # process edge index.
        edge_index = data.edge_index
        edge_index, _ = safe_to_undirected(edge_index)
        data.edge_index = edge_index

        # add edge text:
        edge_text_lst = ["Connected papers are cited together by other papers."]
        edge_map = torch.zeros(edge_index.size(-1), dtype=torch.long)
        data.edge_attr = edge_text_lst
        data.edge_map = edge_map

        # save original feature with _original suffix.
        data.rename_key("x_original", "x")
        data.node_map = torch.arange(len(data.x_original), dtype=torch.long)

        # node text
        node_text_lst = ["Academic paper with title and abstract: " + text for text in
                         data.raw_texts]
        # save raw text of node feature with x
        data.replace_key("x", node_text_lst, "raw_texts")

        # additional label description.
        with open(self.raw_paths[-1]) as f:
            category_desc = json.load(f)

        ordered_desc = BaseDict()
        label_names = []
        for i in range(len(category_desc)):
            label = category_desc[i]["name"]
            label_names.append(label)
            desc = category_desc[i]["description"]
            ordered_desc[label] = desc

        # add link prediction label:
        label_names = label_names + ["No", "Yes"]

        data.replace_key("label", label_names, "label_names")

        data.delete_keys("category_names")
        data.rename_key("label_map", "y")
        side_data = BaseDict(
            node_split=BaseDict({"train": data["train_masks"],
                                 "val": data["val_masks"],
                                 "test": data["test_masks"]}),
            label_description=ordered_desc,
        )

        data.delete_keys(["train_masks", "val_masks", "test_masks"])

        return [data], side_data

    def get_NP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the node-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        mask = self.side_data.node_split[split][0]
        indexs = torch.where(mask)[0]
        labels = self.label_map[indexs]
        label_map = labels
        return indexs, labels, label_map.tolist()

    def get_LP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the link-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        offset = 7
        indexs, labels = self.side_data.link_split[split]
        label_map = labels + offset
        return indexs, labels, label_map.tolist()

    def get_NQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for node question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["What is the most likely paper category for the paper?"]
        answer_list = []
        label_features = self.label[:7]
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]
        return label_map, q_list, a_list

    def get_LQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for link question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["Is two papers co-cited or not? Please answer yes if two papers are co-cited and no otherwise."]
        answer_list = []
        label_features = self.label
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]

        return label_map, q_list, a_list
