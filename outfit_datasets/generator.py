import logging
from typing import Any, Callable
import numpy as np

from . import utils


def _run_unimplemented(self, *input: Any) -> None:
    r"""Generate outfit tuples.

    Should be overridden by all subclasses.
    """
    raise NotImplementedError


class Generator:
    r"""Base class for all generator.
    """

    run: Callable[..., Any] = _run_unimplemented

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def __call__(self, data: np.ndarray = None) -> np.ndarray:
        self.logger.info(f"Generating tuples with {self} mode.")
        return self.run(data)

    def extra_repr(self) -> str:
        """Set the extra representation."""
        return ""

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.extra_repr() + ")"


class FixGenerator(Generator):
    """Always return registered tuples."""

    def __init__(self, data: np.ndarray = None):
        super().__init__()
        self.data = data

    def run(self, *input: Any) -> np.ndarray:
        """Fixed generator.

        Return registed tuples.
        """
        return self.data


class IdentityGenerator(Generator):
    def __init__(self):
        super().__init__()

    def run(self, data: np.ndarray = None) -> np.ndarray:
        """Identity generator.

        Retrun input tuples
        """
        return data


class ResampleGenerator(Generator):
    """Resample subset from outfits."""

    def __init__(self, ratio: float = 0.3):
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


class RandomMixGenerator(Generator):
    def __init__(self, ratio: int = 1, type_aware: bool = False):
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
        item_list, all_items = utils.get_item_list(data)
        self.logger.info("Sampling x%d outfits from set %s", self.ratio, str(list(map(len, item_list))))
        pos_uids, pos_sizes, pos_items, pos_types = utils.split_tuple(data)
        neg_uids = pos_uids.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_sizes = pos_sizes.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_types = pos_types.repeat(self.ratio, axis=0)
        neg_items = []
        pos_set = set(map(tuple, pos_items))
        for item_types in pos_types:
            n_sampled = 0
            while n_sampled < self.ratio:
                if self.type_aware:
                    sampled = [np.random.choice(item_list[i]) for i in item_types]
                else:
                    sampled = [np.random.choice(all_items) for _ in item_types]
                if tuple(sampled) not in pos_set:
                    n_sampled += 1
                    neg_items.append(sampled)
        neg_items = np.array(neg_items)
        return np.hstack([neg_uids, neg_sizes, neg_items, neg_types])

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}, type_aware={self.type_aware}"


class RandomReplaceGenerator(Generator):
    r"""Replace :math:`n` item in outfit."""

    def __init__(self, ratio=1, num_replace=1, type_aware=False):
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
        self.logger.info(
            "Generating tuples with RandomReplace(ratio={}, num_replace={}, type_aware={}) mode.".format(
                self.ratio, self.num_replace, self.type_aware
            )
        )
        data = data.repeat(self.ratio, axis=0)
        uids, pos_sizes, pos_items, pos_types = utils.split_tuple(data)
        item_list, all_items = utils.get_item_list(data)
        pos_set = set(map(tuple, pos_items))
        neg_items = []
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
                            sampled_item = np.random.choice(item_list[target_type])
                        else:
                            sampled_item = np.random.choice(all_items)
                    # replace item
                    items[idx] = sampled_item
            neg_items.append(items)
        neg_items = np.array(neg_items)
        neg_data = np.hstack((uids.reshape((-1, 1)), pos_sizes.reshape((-1, 1)), neg_items, pos_types))
        return neg_data

    def extra_repr(self) -> str:
        return f"ratio={self.ratio}, num_repalce={self.num_replace}, type_aware={self.type_aware}"


def getGenerator(
    mode: str, data: np.ndarray = None, ratio: float = 1.0, type_aware: bool = False, num_replace: int = 1,
) -> Generator:
    r"""Get outfit tuple generator.

    Types of generators:

    - "Fix": return stored tuples.
    - "Identity": return input.
    - "RandomMix": reutrn randomly mixed tuples.
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
    if mode == "Fix":
        assert data is not None
        return FixGenerator(data)
    elif mode == "Identity":
        return IdentityGenerator()
    elif mode == "RandomMix":
        return RandomMixGenerator(ratio=ratio, type_aware=type_aware)
    elif mode == "RandomReplace":
        return RandomReplaceGenerator(raito=ratio, type_aware=type_aware, num_replace=num_replace)
    else:
        raise KeyError
