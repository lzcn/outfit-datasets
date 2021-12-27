import logging
import os

import numpy as np
import torchutils
from torch.utils.data import DataLoader
from torchutils import colour

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
        self.param = param
        LOGGER.info("Loading %s data", colour(param.phase))
        LOGGER.info(
            "DataLoader: batch size (%s), number of workers (%s)", colour(param.batch_size), colour(param.num_workers)
        )
        self.num_users = param.num_users
        if param.dataset.data_mode == "FITB":
            pos_fn = param.pos_fitb_fn
            neg_fn = param.neg_fitb_fn
        else:
            pos_fn = param.pos_fn
            neg_fn = param.neg_fn
        if os.path.exists(pos_fn):
            self.pos_data = np.array(torchutils.io.load_csv(pos_fn, converter=int))
            LOGGER.info("Load positive tuples")
            LOGGER.info("Positive tuples shape: %s", colour(f"{self.pos_data.shape}"))
        else:
            self.pos_data = None
            LOGGER.error("Positive tuples does not exist")
        if os.path.exists(neg_fn):
            self.neg_data = np.array(torchutils.io.load_csv(neg_fn, converter=int))
            LOGGER.info("Load negative tuples")
            LOGGER.info("Nagative tuples shape: %s", colour(f"{self.neg_data.shape}"))
        else:
            self.neg_data = None
            LOGGER.warning("Negative tuples does not exist")
        if param.dataset.data_mode == "FITB":
            self.param.shuffle = False
            self.num_questions = len(self.pos_data)
            if self.neg_data is None:
                self.num_fitb_choices = param.dataset.neg_param.get("ratio", 1) + 1
            else:
                self.num_fitb_choices = len(self.neg_data) // len(self.pos_data) + 1
            LOGGER.info("Summary for fill-in-the-blank data set")
            LOGGER.info("Number of FITB questions: %s", colour(f"{self.num_questions:,}"))
            LOGGER.info("Number of FITB answers: %s", colour(self.param.num_fitb_choices))
        else:
            LOGGER.info("Number of positive outfits: %s", colour(f"{len(self.pos_data):,}"))
        self.datum = getDatum(param)
        self.dataset = getOutfitData(
            datum=self.datum, param=param.dataset, pos_data=self.pos_data, neg_data=self.neg_data
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=param.batch_size,
            num_workers=param.num_workers,
            shuffle=param.shuffle,
            drop_last=param.drop_last,
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
