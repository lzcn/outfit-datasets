import logging
from typing import Any, Callable

import numpy as np

from . import utils

_generator_registry = {}


def _run_unimplemented(self, *input: Any) -> None:
    r"""Generate outfit tuples.

    Should be overridden by all subclasses.
    """
    raise NotImplementedError


class Generator:
    r"""Base class for tuple generator.

    This class is used to generate outfit tuples. Subclasses will be registered by the
    class name.

    Example:

        .. code-block:: python

            class MyGenerator(Generator):
                def run(self, data: np.ndarray = None) -> np.ndarray:
                    // ...
                    return data

            generator = MyGenerator(**kwargs)
            tuples = generator(data)
    """

    run: Callable[..., Any] = _run_unimplemented

    def __init_subclass__(cls):
        super().__init_subclass__()
        _generator_registry[cls.__name__] = cls

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def __call__(self, data: np.ndarray = None) -> np.ndarray:
        self.logger.info(f"Generating tuples with {self} mode.")
        return self.run(data)

    def extra_repr(self) -> str:
        return ""

    def info(self):
        self.logger.info("Generating tuples with {}.".format(self.__repr__()))

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.extra_repr() + ")"


class Fix(Generator):
    r"""Always return saved tuples."""

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__()
        assert data is not None, "data must be provided."
        self.data = data

    def run(self, *input: Any) -> np.ndarray:
        """Fixed generator.

        Return registered tuples.
        """
        return self.data


class Identity(Generator):
    r"""Always return input tuples."""

    def __init__(self, **kwargs):
        super().__init__()

    def run(self, data: np.ndarray = None) -> np.ndarray:
        """Identity generator.

        Return input tuples
        """
        return data


class Resample(Generator):
    r"""Resample a subset of input tuples."""

    def __init__(self, ratio: float = 0.3, **kwargs):
        super().__init__()
        self.ratio = ratio

    def run(self, data: np.ndarray) -> np.ndarray:
        num_outfits = int(len(data) * self.ratio)
        uidxs = data[:, 0]
        user_data = [data[uidxs == u] for u in range(len(set(uidxs)))]
        sampled = np.vstack([data[np.random.choice(len(data), num_outfits, replace=False)] for data in user_data])
        return sampled

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}"


class RandomMix(Generator):
    r"""Return randomly mixed tuples.

    Gian an outfit :math:`\{x_1, \ldots,x_n\}`, for each item :math:`x_i`, randomly sample
    a new item :math:`x_i^-` to get a negative outfit :math:`\{x_1^-, \ldots,x_n^-\}`.

    Args:

        ratio (int): ratio of negative outfits to be sampled for each positive outfit.
        type_aware (bool): whether to sample negative item :math:`x_i^-` with the same
            type of :math:`x_i`.

    """

    def __init__(self, ratio: int = 1, type_aware: bool = False, **kwargs):
        super().__init__()
        self.type_aware = type_aware
        self.ratio = ratio

    def run(self, data: np.ndarray) -> np.ndarray:
        """Random mixing items.

        Args:
            data (np.ndarray): positive tuples

        Returns:
            np.ndarray: negative tuples
        """
        item_list = utils.get_item_list(data)
        num_items = list(map(len, item_list))
        num_types = utils.infer_num_type(data)
        max_items = utils.infer_max_shape(data)
        if self.type_aware:
            self.logger.info("Sampling {}x outfits from {:,} sets: {}".format(self.ratio, len(item_list), num_items))
        else:
            self.logger.info("Sampling {}x outfits from {:,} items".format(self.ratio, np.sum(num_items)))
        pos_uids, pos_sizes, pos_items, pos_types = utils.split_tuple(data)
        neg_uids = pos_uids.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_sizes = pos_sizes.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_types = []
        neg_items = []
        pos_set = set(map(tuple, pos_items))
        for size, item_types in zip(pos_sizes, pos_types):
            n_sampled = 0
            while n_sampled < self.ratio:
                if self.type_aware:
                    sampled_types = item_types
                else:
                    sampled_types = np.random.randint(num_types, size=max_items)
                sampled_items = [np.random.choice(item_list[i]) for i in sampled_types]
                sampled_items = sampled_items[:size] + [utils.NONE_TYPE] * (max_items - size)
                if tuple(sampled_items) not in pos_set:
                    n_sampled += 1
                    neg_items.append(sampled_items)
                    neg_types.append(sampled_types)
        neg_items = np.array(neg_items)
        neg_types = np.array(neg_types)
        return np.hstack([neg_uids, neg_sizes, neg_items, neg_types])

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}, type_aware={self.type_aware}"


class RandomHard(Generator):
    r"""Return Randomly outfits sampled from other users.

    Args:

        ratio (int): ratio of negative outfits to be sampled for each positive outfit.

    """

    def __init__(self, ratio: int = 1, **kwargs):
        super().__init__()
        self.ratio = ratio

    def run(self, data: np.ndarray) -> np.ndarray:
        """Random mixing items.

        Args:
            data (np.ndarray): positive tuples

        Returns:
            np.ndarray: negative tuples
        """
        pos_uids, pos_sizes, pos_items, pos_types = utils.split_tuple(data)
        neg_uids = pos_uids.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_sizes = []
        neg_types = []
        neg_items = []
        num_outfits = len(pos_uids)
        for uid in neg_uids:
            sampled_user = uid
            while sampled_user == uid:
                idx = np.random.randint(num_outfits)
                sampled_user = pos_uids[idx]
            neg_items.append(pos_items[idx])
            neg_types.append(pos_types[idx])
            neg_sizes.append(pos_sizes[idx])
        neg_sizes = np.array(neg_sizes).reshape((-1, 1))
        neg_items = np.array(neg_items)
        neg_types = np.array(neg_types)
        return np.hstack([neg_uids, neg_sizes, neg_items, neg_types])

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}"


class RandomAddon(Generator):
    def __init__(self, ratio: int = 1, **kwargs):
        super().__init__()
        self.ratio = ratio

    def run(self, data: np.ndarray) -> np.ndarray:
        """Random mixing items.

        Args:
            data (np.ndarray): positive tuples

        Returns:
            np.ndarray: negative tuples
        """
        item_list = utils.get_item_list(data)
        num_types = utils.infer_num_type(data)
        pos_uids, pos_sizes, pos_items, pos_types = utils.split_tuple(data)
        neg_uids = pos_uids.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_sizes = pos_sizes.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_sizes += 1
        neg_types = []
        neg_items = []
        pos_set = set(map(tuple, pos_items))
        for items, types in zip(pos_items, pos_types):
            n_sampled = 0
            while n_sampled < self.ratio:
                # sample an item
                sampled_type = np.random.randint(num_types)
                sampled_item = np.random.choice(item_list[sampled_type])
                neg_types.append([sampled_type] + types.tolist())
                neg_items.append([sampled_item] + items.tolist())
                n_sampled += 1
        neg_items = np.array(neg_items)
        neg_types = np.array(neg_types)
        return np.hstack([neg_uids, neg_sizes, neg_items, neg_types])

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}"


class RandomReplace(Generator):
    r"""Replace :math:`n` item in outfit."""

    def __init__(self, ratio=1, num_replace=1, type_aware=False, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.num_replace = num_replace
        self.type_aware = type_aware

    def run(self, data: np.ndarray) -> np.ndarray:
        """Randomly replace :math:`n` items.

        Args:
            data (np.ndarray): positive tuples

        Returns:
            np.ndarray: negative tuples
        """
        self.logger.info("Generating tuples with {}.".format(self.__repr__()))
        data = data.copy().repeat(self.ratio, axis=0)
        uids, pos_sizes, pos_items, pos_types = utils.split_tuple(data)
        item_list = utils.get_item_list(data)
        num_types = utils.infer_num_type(data)
        pos_set = set(map(tuple, pos_items))
        neg_items = []
        neg_types = []
        for size, items, types in zip(pos_sizes, pos_items, pos_types):
            num_replace = min(size, self.num_replace)
            replace_index = np.random.choice(size, num_replace, replace=False)
            while tuple(items) in pos_set:
                for idx in replace_index:
                    target_type = types[idx]
                    target_item = items[idx]
                    # random sample an item
                    sampled_item = target_item
                    while sampled_item == target_item:
                        if self.type_aware:
                            sampled_type = target_type
                            sampled_item = np.random.choice(item_list[target_type])
                        else:
                            sampled_type = np.random.randint(num_types)
                            sampled_item = np.random.choice(item_list[sampled_type])
                    # replace item and type
                    items[idx] = sampled_item
                    types[idx] = sampled_type
            neg_items.append(items)
            neg_types.append(types)
        neg_items = np.array(neg_items)
        neg_types = np.array(neg_types)
        neg_data = np.hstack((uids.reshape((-1, 1)), pos_sizes.reshape((-1, 1)), neg_items, neg_types))
        return neg_data

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}, num_repalce={self.num_replace}, type_aware={self.type_aware}"


class FITB(Generator):
    r"""Replace one item in outfit."""

    def __init__(self, ratio=1, type_aware=False, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.type_aware = type_aware

    def run(self, data: np.ndarray) -> np.ndarray:
        """Randomly replace :math:`n` items.

        Args:
            data (np.ndarray): positive tuples

        Returns:
            np.ndarray: negative tuples
        """
        pos_uidxs, pos_sizes, pos_items, pos_types = utils.split_tuple(data)
        item_list = utils.get_item_list(data)
        num_types = utils.infer_num_type(data)
        neg_uidxs = pos_uidxs.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_sizes = pos_sizes.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_items = []
        neg_types = []
        for size, items, types in zip(pos_sizes, pos_items, pos_types):
            # for each outfit, generated n samples
            replace_index = np.random.choice(size)
            target_type = types[replace_index]
            target_item = items[replace_index]
            item_set = set()
            item_set.add(target_item)
            for _ in range(self.ratio):
                sampled_items = items.copy()
                sampled_types = types.copy()
                sampled_item = target_item
                sampled_type = target_type
                while sampled_item in item_set:
                    # random sample an item
                    if self.type_aware:
                        sampled_type = target_type
                        sampled_item = np.random.choice(item_list[target_type])
                    else:
                        sampled_type = np.random.randint(num_types)
                        sampled_item = np.random.choice(item_list[sampled_type])
                # replace item and type
                sampled_items[replace_index] = sampled_item
                sampled_types[replace_index] = sampled_type
                item_set.add(sampled_item)
                neg_items.append(sampled_items)
                neg_types.append(sampled_types)
        neg_items = np.array(neg_items)
        neg_types = np.array(neg_types)
        neg_data = np.hstack((neg_uidxs, neg_sizes, neg_items, neg_types))
        return neg_data

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}, type_aware={self.type_aware}"


class Retrieval(Generator):
    def __init__(self, ratio=1, type_id=0, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.type_id = type_id

    def run(self, data: np.ndarray) -> np.ndarray:
        """Randomly replace :math:`n` items.

        Args:
            data (np.ndarray): positive tuples

        Returns:
            np.ndarray: negative tuples
        """
        pos_uidxs, pos_sizes, pos_items, pos_types = utils.split_tuple(data)
        item_list = utils.get_item_list(data)
        num_retrieval = len(item_list[self.type_id]) - 1
        neg_uidxs = pos_uidxs.repeat(num_retrieval, axis=0).reshape((-1, 1))
        neg_sizes = pos_sizes.repeat(num_retrieval, axis=0).reshape((-1, 1))
        neg_types = pos_types.repeat(num_retrieval, axis=0)
        neg_items = []
        for items, types in zip(pos_items, pos_types):
            results = np.where(types == self.type_id)[0]
            assert len(results) > 0
            replace_index = results[0]
            assert self.type_id == types[replace_index]
            target_item = items[replace_index]
            item_set = set(item_list[self.type_id])
            item_set.remove(target_item)
            retrieval_items = items.reshape((1, -1)).repeat(num_retrieval, axis=0)
            retrieval_items[:, replace_index] = np.array(list(item_set))
            neg_items.append(retrieval_items)
        neg_items = np.vstack(neg_items)
        neg_data = np.hstack((neg_uidxs, neg_sizes, neg_items, neg_types))
        return neg_data

    def extra_repr(self) -> str:
        return f"type_id={self.type_id}"


def getGenerator(mode: str, data=None, **kwargs) -> Generator:
    r"""Get outfit tuple generator.

    Types of generators:

    - "Fix": return stored tuples.
    - "Identity": return input.
    - "RandomMix": return randomly mixed tuples.
    - "RandomReplace": randomly replace :math:`n` items in outfit.

    Args:
        mode (str): type of generator
        data (np.ndarray, optional): positive or negative tuples. Defaults to None.
        ratio (float, optional): ratio of generated tuples. Defaults to 1.0.
        type_aware (bool, optional): generate tuples while considering item type.
            Defaults to False.
        num_replace (int, optional): number of item replacement. Defaults to 1.

    Returns:
        [Generator]: generator

    """
    supported_modes = ",".join(_generator_registry.keys())
    assert mode in _generator_registry, f"Generator mode {mode} is not support. Only {supported_modes} are supported."
    return _generator_registry[mode](data=data, **kwargs)
