from abc import ABC, abstractmethod
import torch
from torchmetrics import Accuracy
from torchmetrics.classification import AUROC, F1Score, AveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, MeanSquaredLogError, CosineSimilarity, KLDivergence
from torchmetrics.text import BLEUScore, ROUGEScore, Perplexity
from .text_evaluation import TextAccuracy, TextMAE, TextMSE, TextMSLE
from .base_evaluation import MultiApr, MultiAuc
from typing import (
    Optional,
    Callable,
    Union
)
from functools import partial


class Evaluator(ABC):
    """Evaluator generator. Return corresponding evaluator given key and parameters.
    """
    name_to_metric = {
        "accuracy": Accuracy,
        "auc": AUROC,
        "f1": F1Score,
        "precision": AveragePrecision,
        "apr": MultiApr,
        "multiauc": MultiAuc,
        "mae": MeanAbsoluteError,
        "mse": MeanSquaredError,
        "rmse": partial(MeanSquaredError, squared=False),
        "msle": MeanSquaredLogError,
        "cos": CosineSimilarity,
        "kld": KLDivergence,
        "perplexity": Perplexity,
        "rouge": ROUGEScore,
        "bleu": BLEUScore,
        "text_accuracy":  TextAccuracy,
        "text_mae": TextMAE,
        "text_mse": TextMSE,
        "text_msle": TextMSLE,
        "text_rmse": partial(TextMSE, squared=False),
    }
    def __new__(
            cls,
            metric_name: str,
            num_classes: Optional[int] = 2,
            num_labels: Optional[int] = 1,
            mode: Optional[str] = "exact_match",
            negative_match_patterns: Optional[Union[list[str], str]] = None,
            regular_patterns: Optional[Union[list[str], str]] = None,
            n_gram: Optional[int] = 4,
            ignore_index: Optional[int] = None,
            **kwargs):
        if metric_name not in cls.name_to_metric:
            raise ValueError(f"{metric_name} is not supported.")
        if metric_name in ["accuracy", "auc", "precision", "F1"]:
            if num_classes >= 2:
                return cls.name_to_metric[metric_name](task="multiclass", num_classes=num_classes, **kwargs)
            else:
                raise ValueError(f"{metric_name} is not supported for {num_classes} classes.")
        elif metric_name in ["apr", "multiauc"]:
            return cls.name_to_metric[metric_name](num_labels=num_labels, **kwargs)
        elif metric_name == "text_accuracy":
            return cls.name_to_metric[metric_name](mode=mode, negative_match_patterns=negative_match_patterns,
                                                   regular_patterns=regular_patterns, **kwargs)
        elif metric_name == "BLEU":
            return cls.name_to_metric[metric_name](n_gram=n_gram, **kwargs)
        elif metric_name == "perplexity":
            return cls.name_to_metric[metric_name](ignore_index=ignore_index, **kwargs)
        else:
            return cls.name_to_metric[metric_name](**kwargs)




