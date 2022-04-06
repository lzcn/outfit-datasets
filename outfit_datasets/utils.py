import logging
import os
from typing import List

import numpy as np
import torchutils
from torchutils import colour

NONE_TYPE = -1
INDEX_BOUND = 2

LOGGER = logging.getLogger(__name__)


def _back_compatibility(tuples):
    if tuples.shape[1] % 2 == 0:
        return tuples
    else:
        uids = tuples[:, :1]
        item_ids, item_types = np.split(tuples[:, 1:], 2, axis=1)
        length = (item_ids != -1).sum(axis=-1).reshape((-1, 1))
        return np.hstack((uids, length, item_ids, item_types))


def load_outfit_tuples(file, extra_info=""):
    """Load outfit tuples from file."""
    if os.path.exists(file):
        data = np.array(torchutils.io.load_csv(file, converter=int))
        data = _back_compatibility(data)
        LOGGER.info("Load {} tuples with shape: {}".format(colour(extra_info), colour(str(data.shape))))
    else:
        data = None
        LOGGER.warning("The {} tuples does not exist".format(colour(extra_info)))
    return data


def split_tuple(tuples: np.ndarray) -> List[np.ndarray]:
    """Split fashion tuples.

    Args:
        tuples (np.ndarray): outfit tuples

    Returns:
        List[np.ndarray]
    """
    uids = tuples[:, 0]
    length = tuples[:, 1]
    item_ids, item_types = np.split(tuples[:, INDEX_BOUND:], 2, axis=1)
    return uids, length, item_ids, item_types


def infer_max_size(tuples: np.ndarray) -> int:
    """Infer the max size from data.

    Args:
        tuples (np.ndarray): fashion outfit tuples

    Returns:
        int: max size
    """
    max_shape = (tuples.shape[1] - 2) // 2
    max_size = np.max(split_tuple(tuples)[1])
    if max_size != max_shape:
        LOGGER.warn("Tuples is not compact")
    return max_size


def infer_num_type(tuples: np.ndarray) -> int:
    """Infer the number of categories from data

    Args:
        tuples (np.ndarray): fashion outfit tuples

    Returns:
        int: number of categories.
    """
    item_types = set(split_tuple(tuples)[-1].flatten())
    if NONE_TYPE in item_types:
        num_type = len(item_types) - 1
    else:
        num_type = len(item_types)
    return num_type


def get_item_list(data: np.ndarray) -> List[np.ndarray]:
    """Return item list for in each fashion category.

    Append non-fashion category in last.

    Args:
        data (np.ndarray): fashion outfit tuples

    Returns:
        List[np.ndarray]: a list of fashion items
    """
    _, _, item_ids, item_types = split_tuple(data)
    num_list = len(set(item_types.flatten()))
    item_set = [set() for _ in range(num_list)]
    for idxs, types in zip(item_ids, item_types):
        for idx, c in zip(idxs, types):
            item_set[c].add(idx)
    item_list = [np.array(list(s)) for s in item_set]
    return item_list


def rearrange(items, types) -> List[list]:
    new_items, new_types = [], []
    for item_id, item_type in zip(items, types):
        if item_type == NONE_TYPE:
            continue
        new_items.append(item_id)
        new_types.append(item_type)
    while len(new_items) < len(items):
        new_items.append(NONE_TYPE)
        new_types.append(NONE_TYPE)
    return new_items, new_types
