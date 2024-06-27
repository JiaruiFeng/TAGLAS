from typing import (
    Union,
    Optional,
)

import numpy as np
import torch
from scipy.sparse import csr_array
from torch import Tensor, LongTensor
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, coalesce


def edge_index_to_csr_adj(
        edge_index: Tensor,
        num_nodes: Optional[int] = None,
        edge_attr: Optional[Tensor] = None, ) -> csr_array:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_attr is None:
        values = torch.ones(len(edge_index[0]))
    else:
        assert len(edge_attr.size()) == 1
        values = edge_attr

    adj = csr_array((values, (edge_index[0], edge_index[1]),),
                    shape=(num_nodes, num_nodes), )
    return adj

def edge_index_to_sparse_csr(
        edge_index: LongTensor,
        edge_attr: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        bidirectional: bool = False) -> SparseTensor:
    N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
    if edge_attr is None:
        edge_attr = torch.arange(edge_index.size(1))
    else:
        assert len(edge_attr.size()) == 1
    if bidirectional:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, -1 - edge_attr], dim=0)
    whole_adj = SparseTensor.from_edge_index(edge_index, edge_attr, (N, N), is_sorted=False)

    rowptr, col, value = whole_adj.csr()  # convert to csr form
    whole_adj = SparseTensor(rowptr=rowptr, col=col, value=value, sparse_sizes=(N, N), is_sorted=True, trust_data=True)
    return whole_adj


def safe_to_undirected(
        edge_index: LongTensor,
        edge_attr: Optional[Tensor] = None):
    if is_undirected(edge_index, edge_attr):
        return edge_index, edge_attr
    else:
        return to_undirected(edge_index, edge_attr)


def sample_k_hop_subgraph_sparse(
    node_idx: Union[int, list[int], Tensor],
    num_hops: int,
    edge_index: SparseTensor,
    max_nodes_per_hop: int = -1,
):
    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx])
    elif isinstance(node_idx, list):
        node_idx = torch.tensor(node_idx)

    assert isinstance(edge_index, SparseTensor)

    subsets = [node_idx]
    for _ in range(num_hops):
        _, neighbor_idx = edge_index.sample_adj(subsets[-1], -1, replace=False)
        if max_nodes_per_hop > 0:
            if len(neighbor_idx) > max_nodes_per_hop:
                neighbor_idx = neighbor_idx[torch.randperm(len(neighbor_idx))[:max_nodes_per_hop]]
        subsets.append(neighbor_idx)

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:len(node_idx)]

    sub_edges = edge_index[subset, :][:, subset].coo()
    row, col, processed_edge_map = sub_edges
    edge_index = torch.stack([row, col], dim=0)

    node_count = subset.size(0)
    edge_index, processed_edge_map = coalesce(edge_index, processed_edge_map, node_count, node_count, "min")

    return subset, edge_index, inv, processed_edge_map


def k_hop_subgraph(
        node_idx: Union[int, list[int], Tensor],
        num_hops: int,
        edge_index: Tensor,
        max_nodes_per_hop=-1,
        relabel_nodes: bool = False,
        num_nodes: Optional[int] = None,
        directed: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, int):
        node_idx = [node_idx]
    elif isinstance(node_idx, Tensor):
        if len(node_idx.size()) == 0:
            node_idx = [node_idx.tolist()]
        else:
            node_idx = node_idx.tolist()

    subsets = []

    for node in node_idx:
        subsets.append(torch.tensor([node], device=row.device))
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            fringe = col[edge_mask]
            if max_nodes_per_hop > 0:
                if len(fringe) > max_nodes_per_hop:
                    fringe = fringe[torch.randperm(len(fringe))[:max_nodes_per_hop]]
            subsets.append(fringe)

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:len(node_idx)]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = row.new_full((num_nodes,), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask
