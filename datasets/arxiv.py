import json
import os
import os.path as osp
import shutil
from typing import (
    Optional,
    Callable,
)

import numpy as np
import pandas as pd
import torch
from torch import LongTensor
from torch import Tensor

from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import TAGDataset, TAGData, BaseDict
from TAGLAS.utils.graph import safe_to_undirected
from TAGLAS.utils.io import download_url, extract_zip, move_files_in_dir, download_hf_file
from TAGLAS.utils.dataset import generate_link_split_loop

class Arxiv(TAGDataset):
    r"""Arxiv citation network dataset.
    """
    zip_url = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
    node_text_url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv"
    graph_description = "This is a citation network from Arxiv platform focusing on the computer science area. Nodes represent academic papers and edges represent citation relationships. "

    def __init__(self,
                 name: str = "arxiv",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, **kwargs)
        self.side_data.link_split, self.side_data.keep_edges = generate_link_split_loop(self._data.edge_index)
    def raw_file_names(self) -> list:
        return ["nodeidx2paperid.csv.gz", "labelidx2arxivcategeory.csv.gz", "edge.csv.gz",
                "node_year.csv.gz", "node-feat.csv.gz", "node-label.csv.gz", "train.csv.gz", "valid.csv.gz",
                "test.csv.gz", "titleabs.tsv", "arxiv.json"]

    def download(self) -> None:
        _ = download_url(self.node_text_url, self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="arxiv", filename="arxiv.json", local_dir=self.raw_dir)
        path = download_url(self.zip_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        dir_name = osp.join(self.raw_dir, "arxiv")
        move_files_in_dir(osp.join(dir_name, "raw"), self.raw_dir)
        move_files_in_dir(osp.join(dir_name, "mapping"), self.raw_dir)
        move_files_in_dir(osp.join(dir_name, "split/time"), self.raw_dir)
        shutil.rmtree(dir_name)

    def gen_data(self) -> tuple[list[TAGData], None]:
        edge = pd.read_csv(self.raw_paths[2], compression='gzip', header=None).values.T.astype(
            np.int64)  # (2, num_edge) numpy array
        node_feat = pd.read_csv(self.raw_paths[4], compression='gzip', header=None).values
        node_feat = node_feat.astype(np.float32)

        node_year = pd.read_csv(self.raw_paths[3], compression='gzip', header=None).values

        nodeidx2paperid = pd.read_csv(self.raw_paths[0], index_col="node idx")
        titleabs = pd.read_csv(self.raw_paths[-2], sep="\t",
                               names=["paper id", "title", "abstract"], index_col="paper id", )

        titleabs = nodeidx2paperid.join(titleabs, on="paper id")
        text = ("Academic paper. Title: " + titleabs["title"] + ". Abstract: " + titleabs["abstract"])
        node_text_lst = text.values
        node_map = torch.arange(len(node_text_lst), dtype=torch.long)

        label = pd.read_csv(self.raw_paths[5], compression='gzip', header=None).values

        train_idx = torch.from_numpy(
            pd.read_csv(self.raw_paths[6], compression='gzip', header=None).values.T[0]).to(
            torch.long)
        valid_idx = torch.from_numpy(
            pd.read_csv(self.raw_paths[7], compression='gzip', header=None).values.T[0]).to(
            torch.long)
        test_idx = torch.from_numpy(
            pd.read_csv(self.raw_paths[8], compression='gzip', header=None).values.T[0]).to(
            torch.long)

        edge_index = torch.from_numpy(edge).long()
        # edge text
        edge_text_lst = ["The connected two papers have a citation relationship."]
        edge_index, _ = safe_to_undirected(edge_index)
        num_edges = edge_index.size(-1)
        edge_map = torch.zeros([num_edges], dtype=torch.long)

        x_original = torch.from_numpy(node_feat).float()
        node_year = torch.from_numpy(node_year).long()
        x = node_text_lst
        label_map = torch.from_numpy(label).long().squeeze()

        with open(self.raw_paths[-1]) as f:
            label_desc = json.load(f)
        label = [label_desc[i]["name"] for i in range(len(label_desc))]

        ordered_desc = BaseDict()
        for i, l in enumerate(label):
            ordered_desc[l] = label_desc[i]["description"]
        label = label + ["No", "Yes"]
        data = TAGData(x=x, node_map=node_map, edge_index=edge_index, edge_attr=edge_text_lst,
                       edge_map=edge_map, x_original=x_original, label=label, label_map=label_map,
                       node_year=node_year)

        side_data = BaseDict(node_split=BaseDict(
            train=train_idx,
            val=valid_idx,
            test=test_idx
        ),
            label_description=ordered_desc, )

        return [data], side_data

    def get_NP_indexs_labels(self, split: str = "train") -> tuple[LongTensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the node-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        indexs = self.side_data.node_split[split]
        labels = self.label_map[indexs]
        label_map = labels
        return indexs, labels, label_map.tolist()

    def get_LP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the link-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        offset = 40
        indexs, labels = self.side_data.link_split[split]
        label_map = labels + offset
        return indexs, labels, label_map.tolist()

    def get_NQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for node question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["What is the most likely paper category for the target arxiv paper?"]
        answer_list = []
        label_features = self.label
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]

        return label_map, q_list, a_list

    def get_LQA_list(self, label_map, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for link question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = [
            "Is two target papers have citation relationship or not? Please answer yes if two papers has citation relationship and no otherwise."]
        answer_list = []
        label_features = self.label
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [(0, l_idx, a_idx) for l_idx, a_idx in zip(label_map, a_idxs)]

        return label_map, q_list, a_list