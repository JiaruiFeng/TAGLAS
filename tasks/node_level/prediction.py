from TAGLAS.data import TAGDataset
from ..base import DefaultTask, SubgraphTask, DefaultTextTask, SubgraphTextTask
from torch import LongTensor, Tensor

def default_labels(dataset: TAGDataset, split: str) -> tuple[LongTensor, Tensor, list]:
    r"""Obtain node prediction labels from each dataset for the specified split.
        Each dataset should implement get_NP_indexs_labels function.
    Args:
        dataset (TAGDataset): Dataset which implement the get_NP_indexs_labels function.
        split (str): Dataset split.
    """

    sample_indexs, sample_labels, sample_label_maps = dataset.get_NP_indexs_labels(split)
    return sample_indexs, sample_labels, sample_label_maps


class DefaultNPTask(DefaultTask):
    """Whole graph node prediction tasks with original node/edge features.
    """

    def __process_split_and_label__(self):
        sample_indexs, sample_labels, sample_label_maps, = default_labels(self.dataset, self.split)
        return sample_indexs, sample_labels, sample_label_maps


class SubgraphNPTask(SubgraphTask):
    r"""Subgraph node prediction tasks with original node/edge features. For each node sample, generate an ego-subgraph around node.
    """

    def __process_split_and_label__(self):
        sample_indexs, sample_labels, sample_label_maps, = default_labels(self.dataset, self.split)
        return sample_indexs, sample_labels, sample_label_maps


class DefaultTextNPTask(DefaultTextTask):
    r"""Whole graph node prediction tasks with text node/edge features.
    """

    def __process_split_and_label__(self):
        sample_indexs, sample_labels, sample_label_maps, = default_labels(self.dataset, self.split)
        return sample_indexs, sample_labels, sample_label_maps


class SubgraphTextNPTask(SubgraphTextTask):
    r"""Subgraph node prediction tasks with text node/edge features. For each node sample, generate an ego-subgraph around node.
    """

    def __process_split_and_label__(self):
        sample_indexs, sample_labels, sample_label_maps, = default_labels(self.dataset, self.split)
        return sample_indexs, sample_labels, sample_label_maps
