from typing import (
    Union,
    Optional,
)

import numpy as np
import torch
from torch import Tensor
from torch_geometric.loader.dataloader import Collater

from TAGLAS.data import TAGData
from TAGLAS.tasks import GQATask
from TAGLAS.tasks.base import BaseTask
from TAGLAS.tasks.base import QATask


class QATaskCollections():
    r"""Task collection class to combine multiple different question answering tasks together for training and inference.
        Currently, only support QA based tasks.
    """

    def __init__(
            self,
            tasks: Union[Union[QATask, GQATask], list[Union[QATask, GQATask]]],
            data_multiple: Optional[Union[list[float], float]] = None,
    ):

        if isinstance(tasks, BaseTask):
            tasks = [tasks]
        self.tasks = tasks
        self.sizes = np.array([len(t) for t in self.tasks])
        self.num_tasks = len(tasks)
        self.data_multiple = data_multiple
        self.base_collater = Collater(None, None)

    @property
    def data_multiple(self):
        return self._data_multiple

    @data_multiple.setter
    def data_multiple(self, data_multiple):
        data_multiple = data_multiple if data_multiple is not None else 1.0
        if isinstance(data_multiple, float):
            self._data_multiple = np.array([data_multiple for _ in range(self.num_tasks)])
        elif isinstance(data_multiple, int):
            self._data_multiple = np.array([data_multiple for _ in range(self.num_tasks)], dtype=np.int32)
        else:
            assert len(data_multiple) == self.num_tasks
            self._data_multiple = np.array(data_multiple)
        self.compute_sizes()

    def compute_sizes(self):
        if isinstance(self._data_multiple[0], float):
            self.aug_sizes = (self.sizes * np.array(self._data_multiple)).astype(int)
        elif isinstance(self._data_multiple[0], np.int32):
            self.aug_sizes = self._data_multiple
        self.size_seg = np.cumsum(self.aug_sizes)
        self.ind2task = np.arange(len(self.tasks)).repeat(self.aug_sizes)
        # if data_multiple for all datasets are 1.0, don't do random sample
        if np.sum(self._data_multiple == np.array([1.0 for _ in range(self.num_tasks)])) == self.num_tasks:
            self.sample_ind = np.concatenate([np.arange(size) for size in self.sizes], axis=-1).astype(int)
        else:
            self.sample_ind = (np.random.rand(len(self.ind2task)) * self.sizes.repeat(self.aug_sizes)).astype(int)
        self.data_start_index = np.r_[0, self.size_seg[:-1]]

    def __getitem__(self, index):
        task_ind = self.ind2task[index]
        task = self.tasks[task_ind]
        data = task[self.sample_ind[index]]
        data.task_idx = task_ind
        return data

    def __len__(self):
        return np.sum(self.aug_sizes)


    def collate(self, batch: list[TAGData]):
        return self.tasks[0].collate(batch)
