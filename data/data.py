from typing import (Optional,
                    Union,
                    Any)

import numpy as np
from torch import Tensor, LongTensor
from torch_geometric.data import Data

from .base import DictFunctions


class TAGData(Data, DictFunctions):
    r"""
    Data object for saving and processing Text-Attributed Graphs (TAGs) data and related information.
    It inherits from :class:`torch_geometric.data and extend it to support np.ndarray and list input of node and edge text features.
    Args:
        x (Union[list[str], np.ndarray, Tensor], optional): The node feature input to the model. Default arg for saving node text.
        node_map (LongTensor, optional): Mapping from nodes in graph to features.
        edge_index (LongTensor, optional): Graph connectivity in COO format with shape :obj:`[2, num_edges]`. (default: :obj:`None`).
        edge_map (LongTensor, optional): Mapping from edges in graph to edge feature.
        edge_attr (Union[list[str], np.ndarray, Tensor], optional):  The edge feature input to the model. Default arg for saving edge text.
        label (Union[list[str], np.ndarray, Tensor], optional): Store text label set for the graph,
                                could be either node-level, edge-level or graph-level label.
        label_map (LongTensor, optional): Mapping from label of the graph to label feature.
        x_original (Tensor, optional): Original node features (features used in original dataset).
        edge_attr_original (Tensor, optional): Original edge features (features used in original dataset).
    """

    def __init__(self,
                 x: Optional[Union[list[str], np.ndarray]] = None,
                 node_map: Optional[LongTensor] = None,
                 edge_index: Optional[LongTensor] = None,
                 edge_map: Optional[LongTensor] = None,
                 edge_attr: Optional[Union[list[str], np.ndarray, Tensor]] = None,
                 label: Optional[Union[list[str], np.ndarray, Tensor]] = None,
                 label_map: Optional[LongTensor] = None,
                 x_original: Optional[Tensor] = None,
                 edge_attr_original: Optional[Tensor] = None,
                 **kwargs) -> None:
        self.__dict__["_text_keys"] = []
        super().__init__(x, edge_index, edge_attr, **kwargs)
        self.node_map = node_map
        self.edge_map = edge_map
        self.x_original = x_original
        self.edge_attr_original = edge_attr_original
        self.label = label
        self.label_map = label_map

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, 'fset', None) is not None:
            propobj.fset(self, value)
        else:
            setattr(self._store, key, value)
        # For all attributes added into the data, check if its value is text-related.
        if key != "_text_keys":
            self.__check_text_key__(key, value)

    def __check_text_key__(self, key: str, value: Optional[Any] = None):
        if value is None:
            if key in self._text_keys:
                self._text_keys.remove(key)
        else:
            if self.__check_is_text__(value):
                if key not in self._text_keys:
                    self._text_keys.append(key)
            else:
                if key in self._text_keys:
                    self._text_keys.remove(key)

    def __delattr__(self, key: str):
        delattr(self._store, key)
        self.__check_text_key__(key)

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value
        self.__check_text_key__(key, value)

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]
            self.__check_text_key__(key)

    def __check_is_text__(self, content: Any) -> bool:
        if isinstance(content, str) or isinstance(content, np.str_):
            return True
        elif isinstance(content, list):
            if len(content) == 0:
                return False
            return self.__check_is_text__(content[0])
        elif isinstance(content, np.ndarray):
            return self.__check_is_text__(content.tolist())
        else:
            return False

    def text_input_to_numpy_array(self) -> Any:
        r"""Unify all text related input to np.ndarray.
        The text related input are specified in :attr:`text_keys`.
        """
        exist_keys = [key for key in self._store.keys() if key in self._text_keys]
        for key in exist_keys:
            setattr(self, key, self.__text_to_numpy_array__(getattr(self, key)))

        return self

    def __text_to_numpy_array__(
            self,
            input_texts: Union[str, int, float, list[str], np.ndarray]) -> Union[np.ndarray, None]:
        """
        Convert input to np.ndarray format for easy tensor based operations.
        """
        if input_texts is None:
            return input_texts
        elif isinstance(input, str) or isinstance(input_texts, np.str_):
            return np.array([input_texts], dtype=object)
        elif isinstance(input_texts, int) or isinstance(input_texts, float):
            return np.array([str(input_texts)], dtype=object)
        elif isinstance(input_texts, list):
            return np.array(input_texts, dtype=object)
        else:
            return input_texts

    def text_input_to_list(self) -> Any:
        r"""Unify all text related input to list.
        The text related input are specified in :attr:`text_keys`.
        """
        exist_keys = [key for key in self._store.keys() if key in self._text_keys]
        for key in exist_keys:
            setattr(self, key, self.__text_to_list__(getattr(self, key)))

        return self

    def __text_to_list__(
            self,
            input_texts: Union[str, int, float, list[str], np.ndarray]) -> Union[list, None]:
        """
        Convert input to np.ndarray format for easy tensor based operations.
        """
        if input_texts is None:
            return input_texts
        elif isinstance(input, str) or isinstance(input_texts, np.str_):
            return [input_texts]
        elif isinstance(input_texts, int) or isinstance(input_texts, float):
            return [str(input_texts)]
        elif isinstance(input_texts, list):
            return input_texts
        elif isinstance(input_texts, np.ndarray):
            return input_texts.tolist()
        # elif isinstance(input_texts, dict) or isinstance(input_texts, UserDict):
        #     for key, value in input_texts.items():
        #         input_texts[key] = self.__text_to_list__(value)
        else:
            return input_texts
