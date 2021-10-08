from typing import List

import numpy as np
import torch
import torchutils
from torchutils.data import DataReader, getReader

from outfit_datasets.param import OutfitLoaderParam


class Datum(object):
    r"""Wrapper class for datareader.

    The data reader :class:`torchutils.data.DataReader` has the interface
    ``reader(key)`` that returns data of given ``key``.

    Args:
        item_list (List[List]): list for item keys. ``item_list[c][i]`` is the ``key``
            for i-th item in c-th category.
        reader (DataReader): the data reader.
    """

    def __init__(self, item_list: List[List], reader: DataReader):
        self.item_list = item_list
        self.reader = reader

    def get_key(self, item_ids: List, item_types: List, max_size: int = 0) -> List[str]:
        """Return keys for data readerÎ

        Args:
            item_ids (List): item list
            item_types (List): item types
            max_size (int, optional): max size of items. Defaults to 0.

        Returns:
            List[str]: list of item keys
        """
        keys = []
        max_size = max(max_size, len(item_ids))
        for item, cate in zip(item_ids, item_types):
            if cate == -1:
                continue
            else:
                key = self.item_list[cate][item]
                keys.append(key)
        while len(keys) < max_size:
            keys.append(keys[-1])
        return keys

    def get_item(self, item_id: int, item_type: int = None) -> torch.Tensor:
        r"""Get the data of single item.

        Args:
            item_id (int): item id
            item_type (int, optional): item type. Defaults to None.

        Returns:
            torch.Tensor: item data
        """
        key = self.get_key([item_id], [item_type], 1)[0]
        return self.reader(key)

    def get_data(self, item_id: np.ndarray, item_type: np.ndarray = None, max_size: int = 1) -> torch.Tensor:
        r"""Get the data for an outfit

        Args:
            item_id (np.ndarray): item ids
            item_type (np.ndarray, optional): item types. Defaults to None.
            max_size (int, optional): max size. Defaults to 1.

        Returns:
            torch.Tensor: shape of (max_size, *data_shape)
        """
        keys = self.get_key(item_id, item_type, max_size)
        data = torch.stack([self.reader(key) for key in keys], dim=0)
        return data


def getDatum(param: OutfitLoaderParam) -> List[Datum]:
    """Datum factory.

    Args:
        param (DataParam): parameters

    Returns:
        List[Datum]: a list of datum
    """
    datums = []
    item_list = torchutils.io.load_json(param.item_list_fn)
    for reader_param in param.readers:
        reader = getReader(param=reader_param)
        datums.append(Datum(item_list, reader))
    return datums
