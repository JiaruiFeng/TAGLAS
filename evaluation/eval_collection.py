from torchmetrics import Metric
from typing import (
    Union,
    Optional,
)
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class EvaluatorCollection(Metric):
    """Evaluator for task collection.
    Args:
        evaluators (Union[list[nn.Module], nn.Module]): A list of evaluator for each task.
        metric_names (optional, Union[list[str], str]): metric name list for each task.
    """
    def __init__(
            self,
            evaluators: Union[list[Metric], Metric],
            metric_names: Optional[Union[list[str], str]] = None):
        super().__init__()
        if isinstance(evaluators, nn.Module):
            evaluators = [evaluators]
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        assert len(evaluators) == len(metric_names)
        self.evaluators = evaluators
        self.metric_names = metric_names

    def update(
            self, 
            preds: Union[list, Tensor], 
            targets: Union[list, Tensor], 
            task_idxs: Tensor):
        unique_task_idxs = torch.unique(task_idxs)
        for unique_task_idx in unique_task_idxs:
            if isinstance(preds, Tensor):
                task_preds = preds[task_idxs == unique_task_idx]
            else:
                keep_elements = task_idxs == unique_task_idx
                task_preds = []
                for i, pred in enumerate(preds):
                    if keep_elements[i]:
                        task_preds.append(pred)

            if isinstance(targets, Tensor):
                task_targets = targets[task_idxs == unique_task_idx]
            else:
                keep_elements = task_idxs == unique_task_idx
                task_targets = []
                for i, target in enumerate(targets):
                    if keep_elements[i]:
                        task_targets.append(target)
            self.evaluators[unique_task_idx].update(task_preds, task_targets)

    def compute(self):
        result = [evaluator.compute() for evaluator in self.evaluators]
        return dict(zip(self.metric_names, result))

    def reset(self):
        for evaluator in self.evaluators:
            evaluator.reset()

    def to(self, device):
        for evaluator in self.evaluators:
            evaluator.to(device)
