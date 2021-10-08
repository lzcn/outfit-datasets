import logging
import os

import numpy as np
import pandas as pd
import torchutils
from torch.utils.data import DataLoader

from outfit_datasets.dataset import getOutfitData
from outfit_datasets.datum import getDatum
from outfit_datasets.param import OutfitLoaderParam

LOGGER = logging.getLogger(__name__)


class OutfitLoader(object):
    r"""Outfit data class. The class has the following attributes:

    - ``dataset``: :class:`torch.utils.data.Dataset`
    - ``dataloader``: :class:`torch.utils.data.DataLoader`
    - ``num_batch``: number of batches.
    - ``num_sample``: number of samples.

    """

    def __init__(self, param: OutfitLoaderParam = None, **kwargs):
        param = OutfitLoaderParam.evolve(param, **kwargs)
        LOGGER.info("Loading %s data", torchutils.colour(param.phase))
        LOGGER.info(
            "DataLoader: batch size (%s), number of workers (%s)",
            torchutils.colour(param.batch_size),
            torchutils.colour(param.num_workers),
        )
        self.num_users = param.num_users
        self.pos_data = np.array(pd.read_csv(param.pos_fn, dtype=np.int, header=None))
        if os.path.exists(param.neg_fn):
            self.neg_data = np.array(pd.read_csv(param.neg_fn, dtype=np.int, header=None))
            LOGGER.info("Negatives not exists")
        else:
            self.neg_data = None
        if param.dataset.data_mode == "FITB":
            data = np.array(pd.read_csv(param.fitb_fn, dtype=np.int))
            self.pos_data = data
            self.neg_data = None
            self.num_comparisons = len(data)
            LOGGER.info("Summary for fill-in-the-blank data set")
            LOGGER.info("Number of questions: %s", torchutils.colour(self.num_comparisons))
            LOGGER.info("Number of answers: %s", torchutils.colour(self.param.num_choices))
        else:
            LOGGER.info("Number of positive outfits: %d", len(self.pos_data))
        self.datum = getDatum(param)
        self.dataset = getOutfitData(
            datum=self.datum, param=param.dataset, pos_data=self.pos_data, neg_data=self.neg_data
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=param.batch_size,
            num_workers=param.num_workers,
            shuffle=param.shuffle,
            pin_memory=True,
        )

    def build(self):
        """Generate outfits."""
        self.dataset.build()

    def __len__(self):
        return self.num_batch

    @property
    def num_batch(self) -> int:
        try:
            loader = self.dataloader
            return len(loader)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} needs attribute dataloader")

    @property
    def num_sample(self) -> int:
        try:
            dataset = self.dataset
            return len(dataset)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} needs attribute dataset")

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
