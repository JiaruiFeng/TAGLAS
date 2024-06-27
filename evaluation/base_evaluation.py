from torchmetrics import Metric
from torchmetrics.classification import AveragePrecision, AUROC
import torch


class MultiApr(Metric):
    """average precision rate.
    Args:
        num_labels (int): Number of binary labels.
    """
    def __init__(self, num_labels: int = 1, **kwargs):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AveragePrecision(task="binary") for _ in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


class MultiAuc(Metric):
    """average ROC-AUC.
    Args:
        num_labels (int): Number of binary labels.
    """
    def __init__(self, num_labels: int = 1, **kwargs):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AUROC(task="binary") for _ in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()
