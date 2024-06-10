import json
import os.path as osp
from typing import (
    Optional,
    Callable,
    Any
)

import numpy as np
import torch
from torch import Tensor

from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import BaseDict
from TAGLAS.data import TAGDataset, TAGData
from TAGLAS.utils.io import download_hf_file


class FB15K237(TAGDataset):
    """Free base knowledge graph. Contains 237 different relation for link prediction.
    """
    graph_description = "This is a knowledge graph from the FreeBase. Nodes represent knowledge entities and edges represent relations between two entities. "

    def __init__(self,
                 name: str = "fb15k237",
                 root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 to_undirected: bool = True,
                 **kwargs,
                 ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, **kwargs)
        if to_undirected:
            self.to_undirected()

    def to_undirected(self) -> None:
        data = self._data
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_attr_original = data.edge_attr_original
        edge_map = data.edge_map
        keep_edges = self.side_data.keep_edges
        labels = self._data.label

        num_edges = edge_index.size(-1)
        num_edge_type = len(edge_attr)
        edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]])], dim=-1)
        edge_attr = edge_attr + [
            "The inverse relation of "
            + label for label in labels]
        edge_attr_original = torch.cat([edge_attr_original, torch.arange(num_edge_type, num_edge_type * 2)], dim=-1)
        edge_map = torch.cat([edge_map, edge_map + num_edge_type], dim=-1)
        keep_edges = torch.cat([keep_edges, keep_edges + num_edges], dim=-1)
        update_dict = {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_attr_original": edge_attr_original,
            "edge_map": edge_map,
        }
        data.update(update_dict)
        self.data = data
        self.side_data.update(keep_edges=keep_edges)

    def raw_file_names(self) -> list:
        return ["entity2wikidata.json", "train.txt", "valid.txt", "test.txt", "fb15k237.json"]

    def download(self):
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="entity2wikidata.json", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="test.txt", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="train.txt", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="valid.txt", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="FB15K237", filename="fb15k237.json", local_dir=self.raw_dir)

    def gen_data(self) -> tuple[list[TAGData], Any]:

        entity_lst = []
        text_lst = []
        with open(self.raw_paths[0], "r") as f:
            data = json.load(f)

        for k in data:
            # print(data[k])
            entity_lst.append(k)
            text_lst.append("Entity in the knowledge graph. Entity name: " + data[k]["label"]
                            + ", Entity alternatives: " + ", ".join(data[k]["alternatives"]) + ". Entity description:"
                            + data[k]["description"] if data[k]["description"] is not None else "Missing.")

        entity2id = {entity: i for i, entity in enumerate(entity_lst)}

        relation2id = {}
        converted_triplets = {}
        rel_list = []
        rel = len(relation2id)
        names = ["train", "valid", "test"]
        name_dict = {n: osp.join(self.raw_dir, n + ".txt") for n in names}

        for file_type, file_path in name_dict.items():
            edges = []
            edge_types = []
            with open(file_path) as f:
                file_data = [line.split() for line in f.read().split("\n")[:-1]]
            unknown_entity = 0
            for triplet in file_data:
                if triplet[0] not in entity2id:
                    text_lst.append("entity names: Unknown")
                    entity_lst.append(triplet[0])
                    entity2id[triplet[0]] = len(entity2id)
                    unknown_entity += 1
                if triplet[2] not in entity2id:
                    text_lst.append("entity names: Unknown")
                    entity_lst.append(triplet[2])
                    entity2id[triplet[2]] = len(entity2id)
                    unknown_entity += 1
                if triplet[1] not in relation2id:
                    relation2id[triplet[1]] = rel
                    rel_list.append(
                        "Relation from source entity to target entity: " + triplet[1])
                    rel += 1

                edges.append([entity2id[triplet[0]], entity2id[triplet[2]], ])
                edge_types.append(relation2id[triplet[1]])
            converted_triplets[file_type] = [edges, edge_types]

        node_map = torch.arange(len(text_lst), dtype=torch.long)

        edge_index = torch.cat([torch.tensor(e[0]) for e in converted_triplets.values()], dim=0).transpose(0, 1)
        edge_map = torch.cat([torch.tensor(e[1]) for e in converted_triplets.values()], dim=-1)
        num_train, num_val, num_test = (len(converted_triplets["train"][0]), len(converted_triplets["valid"][0]),
                                        len(converted_triplets["test"][0]))

        keep_edges = torch.arange(num_train)
        train_idx = torch.arange(num_train)
        val_idx = torch.arange(num_train, num_val + num_train, )
        test_idx = torch.arange(num_val + num_train, num_train + num_val + num_test)

        with open(self.raw_paths[-1]) as f:
            label_desc = json.load(f)

        ordered_desc = BaseDict()
        labels = []
        for i in range(len(label_desc)):
            label = label_desc[i]["name"]
            labels.append(label)
            desc = label_desc[i]["description"]
            ordered_desc[label] = desc
        edge_attr = ["Relation from source entity to target entity: " + label for label in
                     labels]
        data = TAGData(x=text_lst, node_map=node_map, x_original=torch.zeros([len(text_lst), 1]), edge_attr=edge_attr,
                       edge_attr_original=torch.tensor(list(relation2id.values())), edge_index=edge_index.long(),
                       edge_map=edge_map, label=labels, label_map=edge_map, )

        side_data = BaseDict(link_split=BaseDict(
            train=train_idx,
            val=val_idx,
            test=test_idx),
            keep_edges=keep_edges,
            label_description=ordered_desc, )

        return [data], side_data

    def get_LP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the link-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        idxs = self.side_data.link_split[split]
        edge_index = self.edge_index
        labels = self.label_map[idxs]
        label_map = labels
        idxs = edge_index[:, idxs].transpose(0, 1)
        return idxs, labels, label_map.tolist()

    def get_LQA_list(self, label_map, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for link question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["What is the relationship between two target entities?"]
        answer_list = []
        label_features = self.label
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]

        return label_map, q_list, a_list
