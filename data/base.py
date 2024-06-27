import weakref
from collections import UserDict
from typing import (
    Any,
    Union,
    Optional,
)


class DictFunctions:
    r"""Utility function collection for dict operation.
    """

    def delete_keys(self, keys: Union[str, list[str]]) -> None:
        r"""Delete corresponding keys from the dict.
        Args:
            keys (Union[str, list[str]]): The key to delete.
        """
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            self.__delitem__(key)

    def rename_key(
            self,
            new_key: str,
            old_key: str) -> None:
        r"""Rename the old_key with the new_key in the dict.
        Args:
            new_key (str): The new key.
            old_key (str): The old key.
        """
        self.__setitem__(new_key, self.__getitem__(old_key))
        self.__delitem__(old_key)

    def replace_key(
            self,
            new_key: str,
            new_value: Any,
            old_key: str) -> None:
        r"""Replace the old_key with the new_value and new_key.
        Args:
            new_key (str): The new key.
            new_value (Any): The new value.
            old_key (str): The old key.
        """
        self.__setitem__(new_key, new_value)
        self.__delitem__(old_key)


class BaseDict(UserDict, DictFunctions):
    r"""Base dict class for saving data and attributes.
    Args:
        data (dict, optional): The initial dict. All key-value pair will be added to the initialized dict.
    """

    def __init__(
            self,
            data: Optional[dict] = None,
            **kwargs) -> None:
        super().__init__(data, **kwargs)

    def __getattr__(self, key: str) -> Any:
        if key == 'data':
            return self.__dict__["data"]
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            ) from None

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, 'fset', None) is not None:
            propobj.fset(self, value)
        elif key == '_parent':
            self.__dict__[key] = weakref.ref(value)
        elif key[:1] == '_':
            self.__dict__[key] = value
        elif key == "data":
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delattr__(self, key: str) -> None:
        if key[:1] == '_':
            del self.__dict__[key]
        else:
            del self[key]
