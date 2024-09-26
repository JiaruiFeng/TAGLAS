import gc
import os
import os.path as osp
import shutil
import warnings
from typing import (
    Optional,
    Callable, Any,
)

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import json
from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import TAGDataset, TAGData, BaseDict
from TAGLAS.data.dataset import ROOT
from TAGLAS.utils.graph import safe_to_undirected
from TAGLAS.utils.io import download_url, extract_zip, download_hf_file


class GenNodes(Dataset):
    def __init__(self, edge_index, node_exist_map):
        self.edge_index = edge_index
        self.node_exist_map = node_exist_map

    def __len__(self):
        return self.edge_index.size(-1)

    def __getitem__(self, item):
        source, target = self.edge_index[:, item]
        source = source.item()
        target = target.item()
        if (self.node_exist_map[source] or self.node_exist_map[target]):
            return source, target
        else:
            return None


class GenEdges(Dataset):
    def __init__(self, edge_index, include_nodes):
        self.edge_index = edge_index
        self.include_nodes = include_nodes

    def __len__(self):
        return self.edge_index.size(-1)

    def __getitem__(self, item):
        source, target = self.edge_index[:, item]
        if (source in self.include_nodes and target in self.include_nodes):
            return source, target
        else:
            return None


class MAG240M(TAGDataset):
    data_url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/mag240m_kddcup2021.zip'
    mapping_url = "http://snap.stanford.edu/ogb/data/lsc/mapping/mag240m_mapping.zip"
    graph_description = "This is a citation network from microsoft academic graph platform. Nodes represent academic papers and edges represent citation relationship. "

    def __init__(
            self,
            name: str = "mag240m",
            root: Optional[str] = None,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            subset: bool = True,
            num_workers: int = 128,
            **kwargs,
    ) -> None:
        self.subset = subset
        self.name = name
        self.num_workers = num_workers
        root = (root if root is not None else ROOT)
        root = osp.join(root, self.name)
        super(InMemoryDataset, self).__init__(root, transform, pre_transform, pre_filter)
        update_dict = {}
        key_name_list = ["x", "node_map", "edge_attr", "edge_map", "edge_index", "label", "label_map", "side_data"]
        for i, path in enumerate(self.processed_paths):
            data = torch.load(path)
            update_dict[key_name_list[i]] = data
        data = TAGData(**update_dict)
        self.data, self.slices = self.collate([data])

    def raw_file_names(self) -> list:
        return ["meta.pt", "split_dict.pt", osp.join("mag240m_mapping", "text.csv"),
                osp.join("processed", "paper", "node_label.npy"),
                "mag240m.json",
                osp.join("processed", "paper___cites___paper", "edge_index.npy")]

    def processed_file_names(self) -> list:
        suffix = ""
        if self.subset:
            suffix = "_subset"
        return [f"x{suffix}.pkl", f"node_map{suffix}.pkl", f"edge_attr{suffix}.pkl",
                f"edge_map{suffix}.pkl", f"edge_index{suffix}.pkl", f"label{suffix}.pkl",
                f"label_map{suffix}.pkl", f"side_data{suffix}.pkl"]

    def download(self):
        data_path = download_url(self.data_url, self.root)
        extract_zip(osp.join(self.root, "mag240m_kddcup2021.zip"), self.root)
        os.remove(data_path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, "mag240m_kddcup2021"), self.raw_dir)
        mapping_path = download_url(self.mapping_url, self.raw_dir)
        extract_zip(osp.join(self.raw_dir, "mag240m_mapping.zip"), self.raw_dir)
        os.remove(mapping_path)
        download_hf_file(HF_REPO_ID, subfolder="mag240m", filename="mag240m.json",
                         local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="mag240m", filename="new_split_dict.pt", local_dir=self.raw_dir)

    def generate_subset(self, num_nodes, split, edge_index):
        train, val, test = split["train"], split["valid"], split["test"]
        node_indexs = train.tolist() + val.tolist() + test.tolist()

        node_exist_map = dict(zip(range(num_nodes), [False] * num_nodes))
        for idx in node_indexs:
            node_exist_map[idx] = True

        include_nodes = set()
        include_edge_index = set()
        gen_node_helper = GenNodes(edge_index, node_exist_map)
        loader1 = DataLoader(gen_node_helper, num_workers=self.num_workers, shuffle=False, batch_size=10000,
                             collate_fn=lambda x: x)
        with tqdm(total=len(gen_node_helper), desc="Generate task samples.") as pbar:
            for batch in loader1:
                for item in batch:
                    if item is not None:
                        source, target = item
                        include_nodes.add(source)
                        include_nodes.add(target)
                        include_edge_index.add((source, target))
                pbar.update(len(batch))
        gen_edge_helper = GenEdges(edge_index, include_nodes)
        loader2 = DataLoader(gen_edge_helper, num_workers=self.num_workers, shuffle=False, batch_size=10000,
                             collate_fn=lambda x: x)
        with tqdm(total=len(gen_edge_helper), desc="Generate aux edges.") as pbar:
            for batch in loader2:
                for item in batch:
                    if item is not None:
                        source, target = item
                        include_edge_index.add((source, target))

                pbar.update(len(batch))
        include_nodes = torch.tensor(list(include_nodes), dtype=torch.long)
        include_edge_index = torch.tensor(list(include_edge_index), dtype=torch.long).transpose(0, 1)
        print("Subset node size:", len(include_nodes))
        print("Subset edge size:", include_edge_index.size(-1))
        return include_nodes, include_edge_index

    def gen_data(self) -> tuple[list[TAGData], Any]:
        if not self.subset:
            warnings.warn("Generating full mag240m dataset is not recommended. "
                          "First, the functionality for loading full mag240m is not tested due to the size of dataset."
                          "Meanwhile, the loading will be super slow and require extremely large RAM. ")
        else:
            warnings.warn(
                "Generating mag240m subset. Will use multiprocess with large number of workers for faster process. "
                "You can set num_workers to fit your server.")

        # only include paper2paper relation as we don't have text features for other two node type.
        node_split = torch.load(self.raw_paths[1])
        node_split = BaseDict(**node_split)

        # additional label description.
        with open(self.raw_paths[-1]) as f:
            category_desc = json.load(f)
        label_names = []
        label_text_list = []
        ordered_desc = BaseDict()
        for i in range(len(category_desc)):
            label = category_desc[i]["name"]
            label_names.append(label)
            desc = category_desc[i]["description"]
            ordered_desc[label] = desc

        print("Saving label...")
        torch.save(label_text_list, self.processed_paths[5], pickle_protocol=4)

        del label_text_list
        gc.collect()
        edge_attr = ["Connected two papers have a citation relationship."]
        torch.save(edge_attr, self.processed_paths[2], pickle_protocol=4)

        meta_data = torch.load(self.raw_paths[0])
        num_nodes = meta_data["paper"]
        edge_index = torch.from_numpy(np.load(self.raw_paths[-1]))
        print("begin process edge_index")
        if self.subset:
            subset_node, subset_edge_index = self.generate_subset(num_nodes, node_split, edge_index)
            mapping = edge_index[0].new_full((num_nodes,), -1)
            mapping[subset_node] = torch.arange(subset_node.size(0), device=edge_index.device)
            edge_index = mapping[subset_edge_index]
            train_idx = mapping[node_split["train"]]
            train_idx = train_idx[train_idx != -1]
            node_split["train"] = train_idx
            val_idx = mapping[node_split["valid"]]
            val_idx = val_idx[val_idx != -1]
            node_split["valid"] = val_idx
            test_idx = mapping[node_split["test"]]
            test_idx = test_idx[test_idx != -1]
            node_split["test"] = test_idx

        node_split["val"] = node_split["valid"]
        # node_split.__delitem__("valid")

        edge_index, _ = safe_to_undirected(edge_index)
        print("Saving edge...")
        torch.save(edge_index, self.processed_paths[4], pickle_protocol=4)

        num_edges = edge_index.size(-1)
        edge_map = torch.zeros([num_edges], dtype=torch.long)
        torch.save(edge_map, self.processed_paths[3], pickle_protocol=4)
        del edge_index
        del edge_map
        gc.collect()

        print("begin read node text.")
        node_text_ls = []
        chunksize = 1000000
        total_chunks = num_nodes // chunksize
        chunk_idx = 0
        print(total_chunks)
        for chunk in tqdm(pd.read_csv(self.raw_paths[2], chunksize=chunksize), total=total_chunks):
            print(chunk_idx)
            chunk = chunk.fillna("missing")
            text = (
                    "Academic paper with title and abstract: "
                    + chunk["title"]
                    + ". "
                    + chunk["abstract"]
            )
            node_text_ls.extend(text.tolist())
            chunk_idx += 1
        print("Saving node...")
        if self.subset:
            node_text_ls = [node_text_ls[idx] for idx in subset_node]

        label_map = torch.from_numpy(np.load(self.raw_paths[3])).long()
        label_map = label_map[subset_node]
        torch.save(label_map, self.processed_paths[6], pickle_protocol=4)

        torch.save(node_text_ls, self.processed_paths[0], pickle_protocol=4)
        node_map = torch.arange(len(node_text_ls), dtype=torch.long)
        torch.save(node_map, self.processed_paths[1], pickle_protocol=4)
        del node_text_ls
        del node_map
        gc.collect()

        side_data = BaseDict(node_split=node_split,
                             label_description=ordered_desc)

        return side_data

    def process(self) -> None:
        side_data = self.gen_data()
        if side_data is not None:
            torch.save(side_data, self.processed_paths[-1])
        else:
            torch.save("No side data", self.processed_paths[-1])

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
        q_list = ["What is the most like paper category for the target paper?"]
        answer_list = []
        label_features = self.label
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = np.array(a_list)
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]
        return label_map, q_list, a_list
