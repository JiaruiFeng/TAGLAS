from random import shuffle, randint
from typing import (
    Union,
    Callable,
    Optional,
)

import numpy as np
import torch
from torch import LongTensor, Tensor

from TAGLAS.utils.graph import edge_index_to_csr_adj


def generate_link_split(edge_index: LongTensor, train_ratio: float = 0.85, test_ratio: float = 0.10,
                    labels: Optional[LongTensor] = None) -> tuple[dict, LongTensor]:
    """Random split all links into train/val/test sets. Also sample the equal number of negative links for each split.
    Used if there is no existing split for the given dataset.
    """
    generator = torch.manual_seed(3407)
    num_edges = edge_index.size(1)
    val_ratio = 1.0 - train_ratio - test_ratio
    edge_perm = torch.randperm(num_edges, generator=generator)
    train_offset = int(len(edge_perm) * train_ratio)
    val_offset = int(len(edge_perm) * (train_ratio + val_ratio))
    train_pos_idx, val_pos_idx, test_pos_idx = (
        edge_perm[:train_offset], edge_perm[train_offset:val_offset], edge_perm[val_offset:])
    train_pos_edges, val_pos_edges, test_pos_edges = (
        edge_index[:, train_pos_idx], edge_index[:, val_pos_idx], edge_index[:, test_pos_idx]
    )
    if labels is not None:
        train_label, val_label, test_label = labels[train_pos_idx], labels[val_pos_idx], labels[test_pos_idx]
        link_split = {
            "train": (train_pos_edges.transpose(0, 1), train_label),
            "val": (val_pos_edges.transpose(0, 1), val_label),
            "test": (test_pos_edges.transpose(0, 1), test_label), }
    else:

        # Sample negative edges for training and testing
        adj = edge_index_to_csr_adj(edge_index)
        # Avoid self-edge in negative sampling
        dense_adj = (adj.todense() + np.eye(adj.shape[0])) == 0
        neg_row, neg_col = np.nonzero(dense_adj)
        neg_edge_idx = np.random.permutation(len(neg_row))[: num_edges]
        neg_row, neg_col = neg_row[neg_edge_idx], neg_col[neg_edge_idx]
        neg_edges = torch.from_numpy(np.stack([neg_row, neg_col], axis=1).T).long()
        train_neg_edges, val_neg_edges, test_neg_edges = (
            neg_edges[:, :train_offset], neg_edges[:, train_offset:val_offset], neg_edges[:, val_offset:])
        train_label, val_label, test_label = (
            torch.zeros(train_offset * 2), torch.zeros((val_offset - train_offset) * 2),
            torch.zeros((num_edges - val_offset) * 2))
        train_label[: train_offset] = 1
        val_label[: val_offset - train_offset] = 1
        test_label[: num_edges - val_offset] = 1
        link_split = {"train": (torch.cat([train_pos_edges, train_neg_edges], dim=-1).transpose(0, 1), train_label.long()),
                      "val": (torch.cat([val_pos_edges, val_neg_edges], dim=-1).transpose(0, 1), val_label.long()),
                      "test": (torch.cat([test_pos_edges, test_neg_edges], dim=-1).transpose(0, 1), test_label.long()), }
    return link_split, train_pos_idx.long()


def generate_link_split_loop(edge_index: LongTensor, train_ratio: float = 0.85, test_ratio: float = 0.10) -> tuple[dict, LongTensor]:
    """Random split all links into train/val/test sets. Also sample the equal number of negative links for each split.
    Used if there is no existing split for the given dataset.
    """
    generator = torch.manual_seed(3407)
    num_edges = edge_index.size(1)
    val_ratio = 1.0 - train_ratio - test_ratio
    edge_perm = torch.randperm(num_edges, generator=generator)
    train_offset = int(len(edge_perm) * train_ratio)
    val_offset = int(len(edge_perm) * (train_ratio + val_ratio))
    train_pos_idx, val_pos_idx, test_pos_idx = (
        edge_perm[:train_offset], edge_perm[train_offset:val_offset], edge_perm[val_offset:])
    train_pos_edges, val_pos_edges, test_pos_edges = (
        edge_index[:, train_pos_idx], edge_index[:, val_pos_idx], edge_index[:, test_pos_idx]
    )

    # Sample negative edges for training and testing
    adj = edge_index_to_csr_adj(edge_index)
    n_nodes = adj.shape[0]
    # Avoid self-edge in negative sampling
    rand_edges = torch.randint(n_nodes, (num_edges*2, 2))
    self_edge_mask = rand_edges[:, 0] == rand_edges[:, 1]
    positive_mask = torch.tensor(adj[rand_edges[:,0].tolist(), rand_edges[:,1].tolist()], dtype=torch.long)
    final_mask = torch.logical_and(torch.logical_not(self_edge_mask), torch.logical_not(positive_mask))
    neg_edges = rand_edges[final_mask][:num_edges].T
    train_neg_edges, val_neg_edges, test_neg_edges = (
        neg_edges[:, :train_offset], neg_edges[:, train_offset:val_offset], neg_edges[:, val_offset:])
    train_label, val_label, test_label = (
        torch.zeros(train_offset * 2), torch.zeros((val_offset - train_offset) * 2),
        torch.zeros((num_edges - val_offset) * 2))
    train_label[: train_offset] = 1
    val_label[: val_offset - train_offset] = 1
    test_label[: num_edges - val_offset] = 1
    link_split = {"train": (torch.cat([train_pos_edges, train_neg_edges], dim=-1).transpose(0, 1), train_label.long()),
                  "val": (torch.cat([val_pos_edges, val_neg_edges], dim=-1).transpose(0, 1), val_label.long()),
                  "test": (torch.cat([test_pos_edges, test_neg_edges], dim=-1).transpose(0, 1), test_label.long()), }
    return link_split, train_pos_idx.long()


def generate_sample_split(
        num_samples: int,
        label_map: Optional[Tensor] = None,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2) -> dict:
    """Random split all samples into train/val/test sets. Used if there is no existing split for the given dataset.
    """
    val_ratio = 1.0 - train_ratio - test_ratio
    generator = torch.manual_seed(3407)
    sample_perm = torch.randperm(num_samples, generator=generator)
    train_offset = int(len(sample_perm) * train_ratio)
    val_offset = int(len(sample_perm) * (train_ratio + val_ratio))
    if label_map is None:
        label_map = torch.zeros(num_samples).long()

    sample_split = {
        "train": (sample_perm[:train_offset], label_map[:train_offset]),
        "val": (sample_perm[train_offset:val_offset], label_map[train_offset:val_offset]),
        "test": (sample_perm[val_offset:], label_map[val_offset:]),
    }
    return sample_split


def sample_k_labels_with_true(
        full_label_list: Union[list[str], np.ndarray],
        true_label_idx: int,
        way: int = 10) -> list[str]:
    """
    Sample k labels from a complete label list containing the true label, ensuring that the true label is included
    in the sampled set.

    Parameters:
    full_label_list (list[str]): A complete list of multiple labels.
    true_label (int): The index of the true label.
    way (int, optional): The number of labels to be sampled, defaulting to 10.

    Returns:
    list[str]: A list of sampled labels, guaranteed to contain the true label.
    """

    # Ensure the requested number of labels to sample does not exceed the length of the complete label list
    if way == -1 or len(full_label_list) < way:
        return full_label_list
        # way = len(full_label_list)
    if isinstance(full_label_list, np.ndarray):
        full_label_list = full_label_list.tolist()
    true_label = full_label_list[true_label_idx]
    label_sample_list = full_label_list[:true_label_idx] + full_label_list[true_label_idx + 1:]
    shuffle(label_sample_list)
    label_sample_list = label_sample_list[:way - 1]
    label_sample_list.insert(randint(0, way), true_label)
    shuffle(label_sample_list)
    return label_sample_list

def get_split_data(split: str, split_func: Callable):
    if split == "all":
        splits = ["train", "val", "test"]
        sample_indexs = []
        sample_labels = []
        sample_label_maps = []
        for split in splits:
            indexs, labels, label_maps = split_func(split)
            sample_indexs.append(indexs)
            sample_labels.append(labels)
            sample_label_maps.extend(label_maps)
        sample_indexs = torch.cat(sample_indexs, dim=0)
        sample_labels = torch.cat(sample_labels, dim=0)
    else:
        sample_indexs, sample_labels, sample_label_maps = split_func(split)
    return sample_indexs, sample_labels, sample_label_maps