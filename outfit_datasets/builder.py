import inspect
import logging
from abc import ABCMeta, abstractmethod
from typing import List

import attr
import numpy as np
import torch
from torchutils.param import Param

from outfit_datasets.datum import Datum
from outfit_datasets.generator import getGenerator

from . import utils


@attr.s
class BuilderParam(Param):
    data_mode = attr.ib(default="PairWise")
    posi_mode = attr.ib(default="Fix")
    nega_mode = attr.ib(default="RandomMix")
    posi_param = attr.ib(factory=dict)
    nega_param = attr.ib(factory=dict)


class Builder(metaclass=ABCMeta):
    """Dataset builder.

    The Builder is an abstract factory. Inheritance should implement
    :meth:`process`, :meth:`__getitem__` and :meth:`__len__`. For non-fixed generators,
    rebuild data with :meth:`build` before each epoch.

    Args:
        datum (Datum): a datum reader instance.
        posi_data (nnump.array): positive data
        nega_data (numpy.array, optional): negative data. Defaults to ``None``.
        posi_mode (str, optional): positive generator mode. Defaults to ``"Fix"``.
        nega_mode (str, optional): negative generator mode. Defaults to ``"RandomOnline"``.
        posi_param (GeneratorParam, optional): parameters for positve generator. Defaults to ``None``.
        nega_param (GeneratorParam, optional): parameters for negatie generator. Defaults to ``None``.

    Attributes:
        posi_generator (IGenerator): postive data generator
        nega_generator (IGenerator): negative data negerator
    """

    def __init__(
        self,
        datum: List[Datum],
        posi_data: np.ndarray,
        nega_data=None,
        posi_mode="Fix",
        nega_mode="RandomOnline",
        posi_param=None,
        nega_param=None,
    ):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.datum = datum
        # data format [user, size, items, cates]
        self.num_type = utils.infer_num_type(posi_data)
        self.max_size = utils.infer_max_size(posi_data)
        self.sections = [1, 1, self.max_size, self.max_size]
        self.init_data = posi_data
        self.posi_data = None
        self.nega_data = None
        posi_param = posi_param if posi_param else dict()
        nega_param = nega_param if nega_param else dict()
        self.posi_generator = getGenerator(posi_mode, posi_data, **posi_param)
        self.nega_generator = getGenerator(nega_mode, nega_data, **nega_param)
        self.build()

    def build(self):
        self.logger.info("Generating positive outfits.")
        self.posi_data = self.posi_generator(self.init_data)
        self.logger.info("Positive outfits shape: {}".format(self.posi_data.shape))
        self.logger.info("Generating negative outfits.")
        self.nega_data = self.nega_generator(self.posi_data)
        self.max_size = utils.infer_max_size(self.posi_data)
        self.sections = [1, 1, self.max_size, self.max_size]
        self.process()

    @abstractmethod
    def process(self):
        pass

    def names(self, n):
        pass

    @abstractmethod
    def __getitem__(self, n):
        pass

    @abstractmethod
    def __len__(self):
        pass


class PairwiseOutfitBuilder(Builder):
    def __getitem__(self, n):
        # in current implementation, pos_types == neg_types
        pos_items, pos_types = self.pos_items[n], self.pos_types[n]
        neg_items, neg_types = self.neg_items[n], self.neg_types[n]
        pos_args = (pos_items, pos_types, self.max_size)
        neg_args = (neg_items, neg_types, self.max_size)
        data = []
        for datum in self.datum:
            # m x n x *
            data.append(torch.stack([datum.get_data(*pos_args), datum.get_data(*neg_args)], dim=0))
        data = data[0] if len(self.datum) == 1 else data
        # item category: shape 2 x n
        cate = torch.stack([torch.tensor(pos_types), torch.tensor(neg_types)], dim=0)
        # outfit size: shape 2
        size = torch.tensor([self.pos_len[n], self.neg_len[n]])
        name = (",".join(self.datum[0].get_key(*pos_args)), ",".join(self.datum[0].get_key(*neg_args)))
        return dict(data=data, name=name, size=size, uidx=self.uidxs[n], cate=cate,)

    def __len__(self):
        return len(self.uidxs)

    def process(self):
        # split tuples
        pos_uidx, pos_len, pos_items, pos_types = utils.split_tuple(self.posi_data)
        neg_uidx, neg_len, neg_items, neg_types = utils.split_tuple(self.nega_data)
        # check tuples
        ratio = len(self.nega_data) // len(self.posi_data)
        assert (ratio * len(self.posi_data)) == len(self.nega_data)
        assert (pos_uidx.repeat(ratio, axis=0) == neg_uidx).all()
        # save tuples
        self.uidxs = neg_uidx
        # positive data
        self.pos_len = pos_len.repeat(ratio, axis=0)
        self.pos_items = pos_items.repeat(ratio, axis=0)
        self.pos_types = pos_types.repeat(ratio, axis=0)
        # negative data
        self.neg_len = neg_len
        self.neg_items = neg_items
        self.neg_types = neg_types


class PointwiseOutfitBuilder(Builder):
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
        tuples = np.vstack((self.posi_data, self.nega_data))
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(tuples)
        self.labels = np.array(([1] * len(self.posi_data) + [0] * len(self.nega_data)))


class PositiveOutfitBuilder(PointwiseOutfitBuilder):
    def build(self):
        self.logger.info("Generating positive outfits.")
        self.posi_data = self.posi_generator(self.init_data)
        self.logger.info("Positive outfits shape: {}".format(self.posi_data.shape))
        self.max_size = utils.infer_max_size(self.posi_data)
        self.sections = [1, 1, self.max_size, self.max_size]
        self.process()

    def process(self):
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(self.posi_data)
        self.labels = np.array([1] * len(self.posi_data))


class NegativeOutfitBuilder(PointwiseOutfitBuilder):
    def process(self):
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(self.nega_data)
        self.labels = np.array([0] * len(self.nega_data))


class SequenceOutfitBuilder(PositiveOutfitBuilder):
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


class TripletBuilder(Builder):
    """Triplet daatset.

    Return a triplet (anchor, posi, nega) where posi and nega are from the same category.
    """

    def __init__(
        self, datum: Datum, posi_data: np.ndarray, nega_data, posi_mode, nega_mode, posi_param, nega_param
    ):
        super().__init__(
            datum,
            posi_data,
            nega_data=nega_data,
            posi_mode=posi_mode,
            nega_mode=nega_mode,
            posi_param=posi_param,
            nega_param=nega_param,
        )
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
        _, _, pos_items, pos_types = utils.split_tuple(self.posi_data)
        _, _, neg_items, neg_types = utils.split_tuple(self.nega_data)
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


def getBuilder(
    datum,
    posi_data: np.ndarray,
    nega_data: np.ndarray = None,
    posi_mode: str = "Fix",
    posi_param: dict = None,
    nega_mode: str = "RandomOnline",
    nega_param: dict = None,
    data_mode: str = "PairWise",
) -> Builder:
    _factory = {
        "Positive": PositiveOutfitBuilder,
        "FITB": PositiveOutfitBuilder,
        "Negative": NegativeOutfitBuilder,
        "PointWise": PointwiseOutfitBuilder,
        "PairWise": PairwiseOutfitBuilder,
        "Retrieval": PositiveOutfitBuilder,
        "Sequence": SequenceOutfitBuilder,
        "Triplet": TripletBuilder,
    }
    return _factory[data_mode](
        datum=datum,
        posi_data=posi_data,
        nega_data=nega_data,
        posi_mode=posi_mode,
        nega_mode=nega_mode,
        posi_param=posi_param,
        nega_param=nega_param,
    )
