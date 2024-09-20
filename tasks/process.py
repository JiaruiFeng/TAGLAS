from typing import (
    Union,
    Any,
    Optional
)

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

from TAGLAS.utils.graph import k_hop_subgraph, sample_k_hop_subgraph_sparse
from TAGLAS.utils.io import torch_safe_save, torch_safe_load


def text2feature(
        texts: Union[list[Any], np.ndarray],
        encoder: Any) -> Union[Tensor, list[Tensor]]:
    r"""Encode string collection to a len(data)-by-d matrix, where d is the output dimension of the LLM.
    Args:
        texts (Union[list[Any], np.ndarray]): Collection of texts. Can be list or np.ndarray,
        encoder (Any): Any module that implement an encode function for convert text to embedding.
    """

    if isinstance(texts[0], str):
        if isinstance(texts, np.ndarray):
            return encoder.encode(texts.tolist())
        else:
            return encoder.encode(texts)

    return torch.cat([text2feature(t) for t in texts], dim=0)


def feature_embedding_process(
        texts: Union[list[Any], np.ndarray],
        encoder: Any = None,
        file_name: str = None,
        from_saved: bool = True) -> Tensor:
    """Convert input text features into embedding using the given encoder and save the generated embedding.
    Args:
        texts (Union[list[Any], np.ndarray]): Collection of texts. Can be list or np.ndarray,
        encoder (Any, optional): Any module that implement an encode function for convert text to embedding. Can be None if there exist saved embedding.
        file_name (str, optional): directory for saving and loading embedding. If is None, generate embedding from scratch and do not save it.
        from_saved (bool, opitonal): If true and the file_name if provided, save the generated embedding to the directory specified in file_name.

    """
    if texts is None:
        return None
    if from_saved and file_name is not None:
        embeddings = torch_safe_load(file_name)
        if embeddings is None:
            if encoder is None:
                raise ValueError("There is no saving embedding for the encoder, "
                                 "please initialize corresponding encoder for processing or check the encoder name.")
        else:
            return embeddings
    embeddings = text2feature(texts, encoder)
    if file_name is not None:
        torch_safe_save(embeddings, file_name)
    return embeddings


def subgraph_process(
        index: Union[int, list, Tensor],
        edge_index: Union[LongTensor, SparseTensor],
        node_map: LongTensor,
        edge_map: LongTensor,
        hop: int = 3,
        max_nodes_per_hop: int = -1,
        num_nodes: Optional[int] = None,
        to_sparse: bool = True,
        ppr_scores: Optional[Tensor] = None) -> tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
    """generate subgraph for the input node index.
    """
    if to_sparse:
        subset, processed_edge_index, mapping, processed_edge_map \
            = sample_k_hop_subgraph_sparse(index, hop, edge_index, max_nodes_per_hop, ppr_scores)

    else:
        subset, processed_edge_index, mapping, edge_mask \
            = k_hop_subgraph(index, hop, edge_index, max_nodes_per_hop, True, num_nodes, ppr_scores=ppr_scores)
        processed_edge_map = edge_map[edge_mask]
    processed_node_map = node_map[subset]
    return processed_edge_index, processed_node_map, processed_edge_map, mapping


def value_to_tensor(value: Any, to_long=True):
    r"""Util function to convert all input to tensor and do the uplifting of dimension for 0-dimension tensor.
    """
    if value is None:
        return value
    elif isinstance(value, int) or isinstance(value, np.int64):
        value = torch.tensor([value])
    elif isinstance(value, list):
        value = torch.tensor(value).long()
    elif isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    else:
        if len(value.size()) == 0:
            value = value.unsqueeze(0)
        value = value
    if to_long:
        value = value.long()
    return value


class MultiprocessHelper(Dataset):
    r"""Helper class for using pytorch dataloader multiprocess.
    Args:
        task (Any): Any task module inherit BaseTask.
        graph_level(bool, optional): If true, assume the given task is for graph-level.
    """

    def __init__(self, task: Any, graph_level: bool = False):
        self.task = task
        self.sample_labels = task.sample_labels
        self.sample_indexs = task.sample_indexs
        self.sample_label_map = task.sample_label_map
        self.graph_level = graph_level
        if not graph_level:
            self.edge_index, self.node_map, self.edge_map = self.task.__before_build_dataset__()

    def __getitem__(self, item):
        index = self.sample_indexs[item]
        if self.graph_level:
            edge_index, node_map, edge_map = self.task.__before_build_dataset__(index)
            edge_index = edge_index.clone()
            node_map = node_map.clone()
            edge_map = edge_map.clone()
        else:
            edge_index, node_map, edge_map = self.edge_index, self.node_map, self.edge_map
        y = self.sample_labels[item]
        label_map = self.sample_label_map[item]
        return self.task.__build_sample__(index, y, label_map, edge_index, node_map, edge_map)

    def __len__(self):
        return self.sample_indexs.size(0)


def parallel_build_sample_process(task: Any, graph_level: bool = False):
    r"""Process function for building task with parallel process.
    Args:
        task (Any): Any task module inherit BaseTask.
        graph_level(bool, optional): If true, assume the given task is for graph-level.
    """

    # TODO:Implement own mp process instead of leverage torch dataloader.
    helper = MultiprocessHelper(task, graph_level)
    num_workers = task.num_workers
    num_samples = len(helper)
    batch_size = 100
    data_list = []

    if num_workers > 0:
        sample_per_worker = int(num_samples / num_workers)
        if sample_per_worker >  batch_size:
            mp.set_sharing_strategy('file_system')
            loader = DataLoader(helper, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                collate_fn=lambda x: x)
            with tqdm(total=len(helper), desc="Generate task samples.") as pbar:
                for data in loader:
                    # not sure why need copy to make the process not exceed max_map_count.
                    data = [d.clone() for d in data]
                    data_list.extend(data)
                    pbar.update(len(data))
                    del data
            return data_list

    for i in tqdm(range(len(helper)), total=len(helper), desc="Generate task samples."):
        data_list.append(helper[i])

    return data_list
