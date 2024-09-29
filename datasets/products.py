import json
from copy import deepcopy as c
from typing import (
    Optional,
    Callable, Any,
)

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.utils import to_edge_index

from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import TAGDataset, TAGData, BaseDict
from TAGLAS.utils.io import download_hf_file


class Products(TAGDataset):
    graph_description = "This is a co-purchase network from the Amazon platform. Nodes represent the products sold on Amazon and edges represent two products that are co-purchased together."

    def __init__(self,
                 name: str = "products",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, **kwargs)

    def raw_file_names(self) -> list:
        return ["labelidx2productcategory.csv.gz", "ogbn-products_subset.csv.gz", "ogbn-products_subset.pt",
                "products.json"]

    def download(self) -> None:
        download_hf_file(HF_REPO_ID, subfolder="products", filename="ogbn-products_subset.pt", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="products", filename="labelidx2productcategory.csv.gz",
                         local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="products", filename="ogbn-products_subset.csv.gz",
                         local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="products", filename="products.json", local_dir=self.raw_dir)

    def gen_data(self) -> tuple[list[TAGData], Any]:
        # class 24 contains products with no title and category from different domain.
        # class 44, 45, 46 don't have any sample in this subset (haven't check full data from ogb)
        # remove this two classes from train/val/test mask. replace label with MISSING.
        data = torch.load(self.raw_paths[2])
        node_desc = pd.read_csv(self.raw_paths[1])
        # label_desc = pd.read_csv(osp.join(self.raw_dir, "labelidx2productcategory.csv.gz"), compression='gzip')

        # labels = label_desc["product category"].tolist()
        label_map = data.y.squeeze()
        x_original = data.x
        edge_attr = ["Connected two products are purchased together."]
        node_text_list = []
        node_text_prefix = "Product from Amazon platform with title and content: "

        for i in range(data.num_nodes):
            node_title = (node_desc.iloc[i, 2] if node_desc.iloc[i, 2] is not np.nan else "missing")
            node_content = (node_desc.iloc[i, 3] if node_desc.iloc[i, 3] is not np.nan else "missing")
            text = "Title: " + node_title + ". Content: " + node_content
            node_text_list.append(node_text_prefix + text)

        edge_index = data.adj_t.to_symmetric()
        edge_index = to_edge_index(edge_index)[0]
        train_mask, val_mask, test_mask = data.train_mask.squeeze(), data.val_mask.squeeze(), data.test_mask.squeeze()
        # remove_mask = (data.y != 24).squeeze()
        # train_mask = torch.logical_and(train_mask, remove_mask)
        # val_mask = torch.logical_and(val_mask, remove_mask)
        # test_mask = torch.logical_and(test_mask, remove_mask)

        node_split = BaseDict(train=torch.where(train_mask)[0],
                              val=torch.where(val_mask)[0],
                              test=torch.where(test_mask)[0])



        with open(self.raw_paths[-1]) as f:
            label_desc = json.load(f)

        ordered_desc = BaseDict()
        labels = []
        for i in range(46):
            if i == 24:
                label = "label 25"
                labels.append(label)
                ordered_desc[label] = "MISSING"
            else:
                if i < 24:
                    index = i
                else:
                    index = i - 1
                label = label_desc[index]["name"]
                labels.append(label)
                desc = label_desc[index]["description"]
                ordered_desc[label] = desc
        labels.append("#508510")
        ordered_desc["#508510"] = "MISSING"

        data = TAGData(x=node_text_list,
                       x_original=x_original,
                       node_map=torch.arange(len(node_text_list), dtype=torch.long),
                       edge_index=edge_index,
                       edge_attr=edge_attr,
                       edge_map=torch.zeros(edge_index.size(-1), dtype=torch.long),
                       label=labels,
                       label_map=label_map, )

        side_data = BaseDict(node_split=node_split,
                             label_description=ordered_desc)

        return [data], side_data

    def get_NP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the node-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        indexs = self.side_data.node_split[split]
        labels = self.label_map[indexs]
        label_map = labels
        return indexs, labels, label_map.tolist()

    def get_NQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for node question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["What is the most like category for this product?"]
        answer_list = []
        label_features = c(self.label)
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [(0, l_idx, a_idx) for l_idx, a_idx in zip(label_map, a_idxs)]

        return label_map, q_list, a_list
