from torchmetrics import Metric
from typing import (
    Optional,
    Union,
    Any,
)
import torch
from torch import Tensor
from TAGLAS.utils.text import pattern_exist, extract_numbers, normalize_text, pattern_search
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogError
from torchmetrics.functional.regression.mae import _mean_absolute_error_update
from torchmetrics.functional.regression.mse import _mean_squared_error_update
from torchmetrics.functional.regression.log_mse import _mean_squared_log_error_update



import torch.nn as nn

class TextAccuracy(Metric):
    """Compute accuracy base on text input and prediction.
    Args:
        exact_match (optional, bool): If true, compute accuracy based exact match of the given label. Otherwise, will do word-wise match.
        negative_match_patterns (optional, list): If specified, besides the match of the label,
            input text should not contain patterns in negative_match_patterns(except label).
        search(optional, bool): If true, instead to the match, search on the regular_patterns using regular expression to compute accuracy.
        regular_patterns (optional, list): The search regular expression.
    """

    def __init__(
            self,
            mode: str = "exact_match",
            negative_match_patterns: Optional[Union[list[str], str]] = None,
            regular_patterns: Optional[Union[list[str], str]] = None,
            **kwargs):
        super().__init__()
        self.mode = mode
        self.negative_match_patterns = negative_match_patterns
        if mode == "re":
            assert regular_patterns is not None

        self.regular_patterns = regular_patterns

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @property
    def negative_match_patterns(self):
        return self._negative_match_patterns

    @negative_match_patterns.setter
    def negative_match_patterns(self, value):
        if value is None:
            self._negative_match_patterns = []
        elif isinstance(value, str):
            self._negative_match_patterns = [value]
        elif isinstance(value, list):
            self._negative_match_patterns = value
        else:
            raise ValueError("negative_match_patterns must be a string or a list of strings.")

    @property
    def regular_patterns(self):
        return self._regular_patterns

    @regular_patterns.setter
    def regular_patterns(self, value: Union[str, list[str]]):
        if value is None:
            self._regular_patterns = []
        elif isinstance(value, str):
            self._regular_patterns = [value]
        elif isinstance(value, list):
            self._regular_patterns = value
        else:
            raise ValueError("regular_patterns must be a string or a list of strings.")

    def update(self, preds: list[str], targets: list[str]) -> None:
        if len(preds) != len(targets):
            raise ValueError("preds and target must have the same length.")
        correct = 0
        for pred, target in zip(preds, targets):
            pred_ = normalize_text(pred)
            target_ = normalize_text(target)
            if self.mode == "re":
                for regular_pattern in self.regular_patterns:
                    m = pattern_search(pred_, regular_pattern)
                    if len(m) > 0 and m[0] == target_:
                        correct += 1
                        break
            else:
                if self.mode == "exact_match":
                    if pred_ == target_:
                        correct += 1
                else:
                    if pattern_exist(pred_, target_, exact_match=True):
                        negative_check_count = 0
                        for negative_match_pattern in self.negative_match_patterns:
                            negative_match_pattern_ = normalize_text(negative_match_pattern)
                            if negative_match_pattern_ == target_:
                                negative_check_count += 1
                                continue
                            elif pattern_exist(pred_, negative_match_pattern_, exact_match=True):
                                break
                            else:
                                negative_check_count += 1
                        if negative_check_count == len(self.negative_match_patterns):
                            correct += 1

        self.correct += correct
        self.total += len(targets)

    def compute(self) -> Tensor:
        return self.correct.float() / self.total


def text_to_value(preds: list[str], targets: Union[list[str], Tensor]):
    """
    Extract number from text input and convert to torch tensor for computing metric.
    """
    if len(preds) != len(targets):
        raise ValueError("preds and target must have the same length.")
    value_targets = torch.tensor([float(v) for v in targets])
    value_preds = []
    for pred in preds:
        numbers = extract_numbers(pred)
        value_preds.append(torch.mean(torch.tensor(numbers)).item())
    value_preds = torch.tensor(value_preds)

    return value_preds, value_targets

class TextMAE(MeanAbsoluteError):
    def update(self, preds: list[str], targets: Union[list[str], Tensor]) -> None:
        preds, targets = text_to_value(preds, targets)
        preds = preds.to(self.device)
        targets = targets.to(self.device)
        """Update state with predictions and targets."""
        sum_abs_error, n_obs = _mean_absolute_error_update(preds, targets)

        self.sum_abs_error += sum_abs_error
        self.total += n_obs


class TextMSLE(MeanSquaredLogError):
    def update(self, preds: list[str], targets: Union[list[str], Tensor]) -> None:
        preds, targets = text_to_value(preds, targets)
        preds = preds.to(self.device)
        targets = targets.to(self.device)
        """Update state with predictions and targets."""
        sum_squared_log_error, n_obs = _mean_squared_log_error_update(preds, targets)

        self.sum_squared_log_error += sum_squared_log_error
        self.total += n_obs


class TextMSE(MeanSquaredError):
    def update(self, preds: list[str], targets: Union[list[str], Tensor]) -> None:
        preds, targets = text_to_value(preds, targets)
        preds = preds.to(self.device)
        targets = targets.to(self.device)
        """Update state with predictions and targets."""
        sum_squared_error, n_obs = _mean_squared_error_update(preds, targets, num_outputs=1)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs