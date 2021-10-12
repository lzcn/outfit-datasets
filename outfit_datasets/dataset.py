import logging
from typing import List

import numpy as np
import torch

from outfit_datasets.datum import Datum
from outfit_datasets.generator import Generator, getGenerator
from outfit_datasets.param import OutfitDataParam

from . import utils

_dataset_registry = {}


class BaseOutfitData(object):
    """Base class for dataset.

    Inheritance should implement :meth:`process`, :meth:`__getitem__` and
    :meth:`__len__`.

    Args:
        datum (Datum): a datum reader instance.
        pos_data (np.ndarray): the positive tuples.
        neg_data (np.ndarray, optional): the negative tuples. Defaults to ``None``.
        pos_mode (str, optional): positive generator mode. Defaults to ``"Fix"``.
        neg_mode (str, optional): negative generator mode. Defaults to ``"RandomMix"``.
        pos_param (dict, optional): parameters for positve generator. Defaults to ``None``.
        neg_param (dict, optional): parameters for negatie generator. Defaults to ``None``.

    """

    #: generator for positive tuples
    pos_generator: Generator = None
    #: generator for negative tuples
    neg_generator: Generator = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        _dataset_registry[cls.__name__] = cls

    def __init__(self, datum: List[Datum], param: OutfitDataParam, pos_data: np.ndarray, neg_data: np.ndarray = None):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.datum = datum
        self.num_type = utils.infer_num_type(pos_data)
        self.max_size = utils.infer_max_size(pos_data)
        self.sections = [1, 1, self.max_size, self.max_size]
        self.param = param
        self.pos_data = None
        self.neg_data = None
        self.ini_data = pos_data
        self.pos_generator = getGenerator(self.param.pos_mode, pos_data, **param.pos_param)
        self.neg_generator = getGenerator(self.param.neg_mode, neg_data, **param.neg_param)
        self.build()

    def build(self):
        self.logger.info("Generating positive tuples.")
        self.pos_data = self.pos_generator(self.ini_data)
        self.logger.info("Positive tuples shape: {}".format(self.pos_data.shape))
        self.logger.info("Generating negative tuples.")
        self.neg_data = self.neg_generator(self.pos_data)
        self.logger.info("Nagative tuples shape: {}".format(self.pos_data.shape))
        self.max_size = utils.infer_max_size(self.pos_data)
        self.sections = [1, 1, self.max_size, self.max_size]
        self.process()

    def process(self):
        """Prepare tuples for one epoch.
        """
        raise NotImplementedError

    def names(self, n):
        pass

    def __getitem__(self, n):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class PairwiseOutfit(BaseOutfitData):
    def __getitem__(self, n):
        # in current implementation, pos_types == neg_types
        pos_items, pos_types = self.pos_items[n], self.pos_types[n]
        neg_items, neg_types = self.neg_items[n], self.neg_types[n]
        pos_args = (pos_items, pos_types, self.max_size)
        neg_args = (neg_items, neg_types, self.max_size)
        data = []
        if len(self.datum) == 1:
            datum = self.datum[0]
            data = torch.stack([datum.get_data(*pos_args), datum.get_data(*neg_args)], dim=0)
        else:
            data = [torch.stack([datum.get_data(*pos_args), datum.get_data(*neg_args)], dim=0) for datum in self.datum]
        # item category: shape 2 x n
        cate = torch.stack([torch.tensor(pos_types), torch.tensor(neg_types)], dim=0)
        # outfit size: shape 2
        size = torch.tensor([self.pos_sizes[n], self.neg_sizes[n]])
        name = (",".join(self.datum[0].get_key(*pos_args)), ",".join(self.datum[0].get_key(*neg_args)))
        return dict(data=data, name=name, size=size, uidx=self.uidxs[n], cate=cate)

    def __len__(self):
        return len(self.uidxs)

    def process(self):
        # split tuples
        pos_uidx, pos_sizes, pos_items, pos_types = utils.split_tuple(self.pos_data)
        neg_uidx, neg_sizes, neg_items, neg_types = utils.split_tuple(self.neg_data)
        # check tuples
        ratio = len(self.neg_data) // len(self.pos_data)
        assert (ratio * len(self.pos_data)) == len(self.neg_data)
        assert (pos_uidx.repeat(ratio, axis=0) == neg_uidx).all()
        # save tuples
        self.uidxs = neg_uidx
        # positive data
        self.pos_sizes = pos_sizes.repeat(ratio, axis=0)
        self.pos_items = pos_items.repeat(ratio, axis=0)
        self.pos_types = pos_types.repeat(ratio, axis=0)
        # negative data
        self.neg_sizes = neg_sizes
        self.neg_items = neg_items
        self.neg_types = neg_types


class SubsetData(PairwiseOutfit):
    def __getitem__(self, n):
        # in current implementation, pos_types == neg_types
        item_size = self.pos_sizes[n]
        index = np.random.randint(item_size)
        sub_items, sub_types = list(self.pos_items[n]), list(self.pos_types[n])
        pos_item = sub_items.pop(index)
        pos_type = sub_types.pop(index)
        neg_item, neg_type = self.neg_items[n][index], self.neg_types[n][index]
        if len(self.datum) == 1:
            datum = self.datum[0]
            sub_data = datum.get_data(sub_items, sub_types, self.max_size - 1)
            pos_data = datum.get_item(pos_item, pos_type)
            neg_data = datum.get_item(neg_item, neg_type)
        else:
            # m x n x *
            sub_data = [datum.get_data(sub_items, sub_types, self.max_size - 1) for datum in self.datum]
            pos_data = [datum.get_item(pos_item, pos_type) for datum in self.datum]
            neg_data = [datum.get_item(neg_item, neg_type) for datum in self.datum]
        # item category: shape 2 x n
        cate = torch.tensor(sub_types)
        # outfit size: shape 2
        size = torch.tensor([self.pos_sizes[n], self.neg_sizes[n]])
        return dict(
            data=sub_data,
            size=size,
            uidx=self.uidxs[n],
            cate=cate,
            pos_data=pos_data,
            neg_data=neg_data,
            pos_cate=pos_type,
            neg_cate=neg_type,
        )


class PointwiseOutfit(BaseOutfitData):
    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        # m x n x data_shape
        data = [datum.get_data(items, types, self.max_size) for datum in self.datum]
        data = data[0] if len(self.datum) == 1 else data
        cate = torch.tensor(types)
        return dict(
            size=self.sizes[n],
            label=self.labels[n],
            uidx=self.uidxs[n],
            data=data,
            name=",".join(self.datum[0].get_key(items, types, self.max_size)),
            cate=cate,
        )

    def __len__(self):
        return len(self.uidxs)

    def process(self):
        tuples = np.vstack((self.pos_data, self.neg_data))
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(tuples)
        self.labels = np.array(([1] * len(self.pos_data) + [0] * len(self.neg_data)))


class PositiveOutfit(PointwiseOutfit):
    def build(self):
        self.logger.info("Generating positive tuples.")
        self.pos_data = self.pos_generator(self.ini_data)
        self.logger.info("Positive tuples shape: {}".format(self.pos_data.shape))
        self.max_size = utils.infer_max_size(self.pos_data)
        self.sections = [1, 1, self.max_size, self.max_size]
        self.process()

    def process(self):
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(self.pos_data)
        self.labels = np.array([1] * len(self.pos_data))


class NegativeOutfit(PointwiseOutfit):
    def process(self):
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(self.neg_data)
        self.labels = np.array([0] * len(self.neg_data))


class FITB(PointwiseOutfit):
    def build(self):
        self.pos_data = self.ini_data
        self.max_size = utils.infer_max_size(self.pos_data)
        self.sections = [1, 1, self.max_size, self.max_size]
        self.process()

    def process(self):
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(self.pos_data)
        self.labels = np.array([1] * len(self.pos_data))


class SequenceOutfit(PositiveOutfit):
    """Sequence builder."""

    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        data = self.datum[0].get_data(items, types, self.max_size)
        data = torch.stack(data, dim=0)
        cate = torch.tensor(types)
        return dict(
            size=self.sizes[n],
            label=self.labels[n],
            uidx=self.uidxs[n],
            data=data,
            name=",".join(self.datum[0].get_key(items, types, self.max_size)),
            cate=cate,
        )


class TripletBuilder(BaseOutfitData):
    """Triplet daatset.

    Return a triplet (anchor, posi, nega) where posi and nega are from the same category.
    """

    def __init__(self, datum: List[Datum], param: OutfitDataParam, pos_data: np.ndarray, neg_data: np.ndarray = None):
        super().__init__(datum, param, pos_data, neg_data=neg_data)
        self.logger.info("Building conditions.")
        # build conditions for type-pair
        indx, indy = np.triu_indices(self.num_type, k=1)
        anc_type = np.hstack((indx, indy))
        cmp_type = np.hstack((indy, indx))
        conditions = dict()
        for i, j in zip(anc_type, cmp_type):
            conditions[(i, j)] = len(conditions)
        self.condtions = conditions

    def __getitem__(self, n):
        triplet = []
        for datum in self.datum:
            anc_data = datum.get_data(item_id=self.anc_idx[n], item_type=self.anc_type[n])
            pos_data = datum.get_data(item_id=self.pos_idx[n], item_type=self.cmp_type[n])
            neg_data = datum.get_data(item_id=self.neg_idx[n], item_type=self.cmp_type[n])
            triplet.append((anc_data, pos_data, neg_data))
        triplet = triplet[0] if len(self.datum) == 1 else triplet
        # get triplet
        # conditions are the same for triplet since:
        # 1. pos and neg are from the same category
        # 2. only anc->pos and anc->neg pairs are considered
        if (self.anc_type[n], self.cmp_type[n]) not in self.condtions:
            condition = 0
        else:
            condition = self.condtions[(self.anc_type[n], self.cmp_type[n])]
        types = torch.LongTensor([self.anc_type[n], self.cmp_type[n]])
        return dict(data=triplet, condition=condition, label=-1, types=types)

    def __len__(self):
        return len(self.anc_idx)

    def process(self):
        # split tuples
        _, _, pos_items, pos_types = utils.split_tuple(self.pos_data)
        _, _, neg_items, neg_types = utils.split_tuple(self.neg_data)
        # check tuples
        assert len(pos_items) == len(neg_items)
        assert (pos_types == neg_types).all()
        indx, indy = np.triu_indices(self.max_size, k=1)
        i = np.hstack((indx, indy))
        j = np.hstack((indy, indx))
        # triplet (xi, xj, xk) where xj and xk are from the same category
        anc_item = pos_items[:, i].flatten()
        pos_item = pos_items[:, j].flatten()
        neg_item = neg_items[:, j].flatten()
        anc_type = pos_types[:, i].flatten()
        cmp_type = pos_types[:, j].flatten()
        # check validation
        valid_items = (anc_type != -1) * (cmp_type != -1)
        valid_pairs = (anc_item != pos_item) * (anc_item != neg_item) * (pos_item != neg_item)
        valid = (valid_items * valid_pairs).nonzero()
        self.anc_idx = anc_item[valid]
        self.pos_idx = pos_item[valid]
        self.neg_idx = neg_item[valid]
        self.anc_type = anc_type[valid]
        self.cmp_type = cmp_type[valid]


def getOutfitData(
    datum: List[Datum], param: OutfitDataParam, pos_data: np.ndarray, neg_data: np.ndarray = None
) -> BaseOutfitData:
    """[summary]

    Args:
        datum (List[Datum]): a list of feature readers.
        param (DataBuilderParam): dataset parameter
        pos_data (np.ndarray): positive tuples.
        neg_data (np.ndarray, optional): negative tuples. Defaults to None.

    Returns:
        BaseData: dataset
    """
    return _dataset_registry[param.data_mode](datum, param, pos_data, neg_data)
