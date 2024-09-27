import os
import os.path as osp
import shutil
from typing import (
    Optional,
    Callable, Any,
)

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from TAGLAS.constants import HF_REPO_ID
from TAGLAS.data import TAGDataset, TAGData, BaseDict
from TAGLAS.utils.dataset import generate_link_split
from TAGLAS.utils.io import download_url, extract_zip, download_hf_file


class ML1M(TAGDataset):
    MOVIE_HEADERS = ["movieId", "title", "genres"]
    USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
    RATING_HEADERS = ['userId', 'movieId', 'rating', 'timestamp']
    data_url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    graph_description = "This is a recommendation graph from a movie rating platform. Nodes represent the user or movie in the platform and edges represent the rating a user gives to a movie. "

    def __init__(
            self,
            name="ml1m",
            root: Optional[str] = None,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            to_undirected: Optional[bool] = True,
            **kwargs,
    ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, **kwargs)
        self.side_data.link_split, self.side_data.keep_edges = generate_link_split(self.edge_index, labels=self._data.label_map)

        if to_undirected:
            self.to_undirected()

    def to_undirected(self) -> None:
        resverse_edge_text = "Source movie rated by the target user with rating: "
        data = self._data
        edge_index = data.edge_index
        num_edges = edge_index.size(-1)
        edge_attr = data.edge_attr
        num_edge_attr = len(edge_attr)
        edge_map = data.edge_map
        keep_edges = self.side_data.keep_edges

        label = data.label
        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row]).long()], dim=-1)
        edge_attr_reverse = [resverse_edge_text + i for i in label]
        edge_attr = edge_attr + edge_attr_reverse
        edge_map = torch.cat([edge_map, edge_map + num_edge_attr], dim=-1)
        keep_edges = torch.cat([keep_edges, keep_edges + num_edges], dim=-1)
        update_dict = {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_map": edge_map,
        }
        self._data.update(update_dict)
        self.side_data.keep_edges = keep_edges

    def raw_file_names(self) -> list:
        return ['movies.dat', 'users.dat', 'ratings.dat', "occupation.csv"]

    def download(self):
        path = download_url(self.data_url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-1m')
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)
        download_hf_file(HF_REPO_ID, subfolder="ml1m", filename="occupation.csv", local_dir=self.raw_dir)

    def gen_data(self) -> tuple[list[TAGData], Any]:
        # Process movie data:
        movie_df = pd.read_csv(
            self.raw_paths[0],
            sep='::',
            header=None,
            index_col='movieId',
            names=self.MOVIE_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )

        movie_texts = []
        movie_prefix = "Movie with title and genre. "
        for i in range(movie_df.shape[0]):
            title = movie_df.iloc[i]['title']
            genres = movie_df.iloc[i]['genres']
            movie_texts.append(movie_prefix + "Title: " + title + ". Genre: " + genres)
        num_of_movie = len(movie_texts)
        movie_mapping = {idx: i for i, idx in enumerate(movie_df.index)}

        # Process user data:
        user_df = pd.read_csv(
            self.raw_paths[1],
            sep='::',
            header=None,
            index_col='userId',
            names=self.USER_HEADERS,
            dtype='str',
            encoding='ISO-8859-1',
            engine='python',
        )
        occupation_desc = pd.read_csv(self.raw_paths[-1], header=None, index_col=0)

        user_texts = []
        user_prefix = "User in the movie rating platform with the following information: "
        for i in range(user_df.shape[0]):
            gender = ('man' if user_df.iloc[i]['gender'] == 'M' else 'woman')
            age = str(user_df.iloc[i]['age'])
            occupation = occupation_desc.iloc[int(user_df.iloc[i]['occupation']), 0]
            user_texts.append(user_prefix + "gender: " + gender + ", age: " + age + ", occupation: " + occupation)
        user_mapping = {idx: i for i, idx in enumerate(user_df.index)}

        num_of_user = len(user_texts)
        node_map = torch.arange((num_of_movie + num_of_user))
        node_texts = movie_texts + user_texts

        # Process rating data:
        rating_df = pd.read_csv(
            self.raw_paths[2],
            sep='::',
            header=None,
            names=self.RATING_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )

        label = ["1", "2", "3", "4", "5"]
        edge_text = "Source user rate the target movie with rating: "
        col = torch.tensor([movie_mapping[idx] for idx in rating_df['movieId']], dtype=torch.long)
        row = torch.tensor([user_mapping[idx] for idx in rating_df['userId']], dtype=torch.long) + num_of_movie
        rating = rating_df["rating"]
        label_map = (rating - 1).tolist()
        label_map = torch.tensor(label_map, dtype=torch.long)

        edge_index = torch.stack([row, col])
        edge_attr = [edge_text + i for i in label]

        data = TAGData(x=node_texts,
                       node_map=node_map,
                       label=label,
                       label_map=label_map,
                       edge_index=edge_index,
                       edge_attr=edge_attr,
                       edge_map=label_map)

        return [data], BaseDict()

    def get_LP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the link-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        indexs, labels = self.side_data.link_split[split]
        label_map = labels
        return indexs, labels, label_map.tolist()

    def get_LQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for link question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = ["Please predict the user taste to this movie, ranging from 1 to 5."]

        answer_list = []
        label_features = self.label
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]
        return label_map, q_list, a_list


class ML1M_CLS(ML1M):
    def __init__(
            self,
            name="ml1m_cls",
            root: Optional[str] = None,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            to_undirected: Optional[bool] = True,
            threshold: Optional[int] = 4,
    ) -> None:
        super().__init__(name, root, transform, pre_transform, pre_filter, to_undirected)
        self.threshold = threshold
        reg_label = self.label
        reg_label_map = self.label_map
        cls_label = ["No", "Yes"]
        cls_label_map = torch.tensor([0 if int(reg_label[l]) < self.threshold else 1 for l in reg_label_map],
                                     dtype=torch.long)
        self._data.update(
            {"label": cls_label, "label_map": cls_label_map, "reg_label": reg_label, "reg_label_map": reg_label_map})

    def get_LP_indexs_labels(self, split: str = "train") -> tuple[Tensor, Tensor, list]:
        r"""Return sample labels and their corresponding index for the link-level tasks and the given split.
        Args:
            split (str, optional): Split to use. Defaults to "train".
        """
        indexs, labels = self.side_data.link_split[split]
        label_map = self.label_map[indexs]
        return indexs, label_map, label_map.tolist()

    def get_LQA_list(self, label_map: list, **kwargs) -> tuple[list[list], np.ndarray, np.ndarray]:
        r"""Return question and answer list for link question answering tasks.
        Args:
            label_map (list): Mapping to the label for all samples. Will use it to generate answer and question.
            **kwargs: Other arguments.
        """
        q_list = [
            f"Ranging from 1 to 5, will the user rate the movie with score larger or equal than {str(self.threshold)}? Please answer yes or no."]

        answer_list = []
        label_features = self.label
        for l in label_map:
            answer_list.append(label_features[l] + ".")
        a_list, a_idxs = np.unique(np.array(answer_list, dtype=object), return_inverse=True)
        a_list = a_list.tolist()
        label_map = [[0, l_idx, a_idx] for l_idx, a_idx in zip(label_map, a_idxs)]

        return label_map, q_list, a_list
