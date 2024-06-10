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


def textualize_graph(data):
    # mapping from object id to index
    objectid2nodeid = {object_id: idx for idx, object_id in enumerate(data['objects'].keys())}
    nodes = []
    edges = []
    for objectid, object in data['objects'].items():
        # nodes
        node_attr = f'Object in an image. Name: {object["name"]}'
        x, y, w, h = object['x'], object['y'], object['w'], object['h']
        if len(object['attributes']) > 0:
            node_attr = node_attr + '; attribute: ' + (', ').join(object["attributes"])
        node_attr += '; (x,y,w,h): ' + str((x, y, w, h))
        nodes.append({'node_id': objectid2nodeid[objectid], 'node_attr': node_attr})

        # edges
        for rel in object['relations']:
            src = objectid2nodeid[objectid]
            dst = objectid2nodeid[rel['object']]
            edge_attr = rel['name']
            edge_attr = "Relation between two objects: " + edge_attr
            edges.append({'src': src, 'edge_attr': edge_attr, 'dst': dst})

    return pd.DataFrame(nodes, columns=['node_id', 'node_attr']), pd.DataFrame(edges,
                                                                               columns=['src', 'edge_attr', 'dst'])


class SceneGraph(TAGDataset):
    r"""
    Scene graph dataset.
    """
    graph_description = "This is a scene graph generated from an image. Nodes represent an object in the image and edges represent the relationship between two objects. "

    def __init__(self,
                 name: str = "scene_graph",
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

    def download(self):
        download_hf_file(HF_REPO_ID, subfolder="scenegraph", filename="sceneGraphs.zip", local_dir=self.root)
        extract_zip(osp.join(self.root, "sceneGraphs.zip"), self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="scenegraph", filename="scene_graph_split.pt", local_dir=self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="scenegraph", filename="questions.csv", local_dir=self.raw_dir)

    def gen_data(self) -> tuple[list[TAGData], Any]:
        dataset = json.load(open(self.raw_paths[0]))
        question_df = pd.read_csv(self.raw_paths[1])
        node_txt_list = []
        edge_txt_list = []
        question_txt_list = []
        label_txt_list = []
        answer_txt_list = []
        graphs = []
        image_ids = question_df.image_id.unique()

        for image_id in tqdm(image_ids):
            object_ = dataset[str(image_id)]
            nodes, edges = textualize_graph(object_)
            x = nodes.node_attr.tolist()
            edge_attr = edges.edge_attr.tolist()
            node_map = torch.arange(len(node_txt_list), len(node_txt_list) + len(x))
            node_txt_list.extend(x)
            edge_map = torch.arange(len(edge_txt_list), len(edge_txt_list) + len(edge_attr))
            edge_txt_list.extend(edge_attr)
            edge_index = torch.tensor([edges.src, edges.dst]).long()

            question_text = question_df[question_df.image_id == image_id]["question"].tolist()
            question_map = torch.arange(len(question_txt_list), len(question_txt_list) + len(question_text))
            question_txt_list.extend(question_text)

            label_text = question_df[question_df.image_id == image_id]["answer"].tolist()
            label_map = torch.arange(len(label_txt_list), len(label_txt_list) + len(label_text))
            label_txt_list.extend(label_text)

            answer_text = question_df[question_df.image_id == image_id]["full_answer"].tolist()
            answer_map = torch.arange(len(answer_txt_list), len(answer_txt_list) + len(answer_text))
            answer_txt_list.extend(answer_text)

            graphs.append((image_id, node_map, edge_map, edge_index, question_map, label_map, answer_map))

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
        for image_id, node_map, edge_map, edge_index, question_map_list, label_map_list, answer_map_list in graphs:
            for question_map, label_map, answer_map in zip(question_map_list, label_map_list, answer_map_list):
                data_list.append(
                    TAGData(node_map=node_inverse_map[node_map],
                            edge_index=edge_index,
                            edge_map=edge_inverse_map[edge_map],
                            label_map=label_inverse_map[label_map],
                            question_map=question_inverse_map[question_map],
                            answer_map=answer_inverse_map[answer_map]
                            )
                )
                id_list.append(image_id)

        id_list = torch.tensor(id_list)
        graph_split = torch.load(self.raw_paths[2])
        train_mask = torch.tensor([True if id in graph_split["train"] else False for id in id_list])
        val_mask = torch.tensor([True if id in graph_split["val"] else False for id in id_list])
        test_mask = torch.tensor([True if id in graph_split["test"] else False for id in id_list])
        train_idx = torch.where(train_mask)[0]
        val_idx = torch.where(val_mask)[0]
        test_idx = torch.where(test_mask)[0]
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
