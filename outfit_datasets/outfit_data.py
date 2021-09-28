import logging

import numpy as np
import pandas as pd
import torchutils
from torch.utils.data import DataLoader
import attr
from outfit_datasets.builder import getBuilder
from outfit_datasets.data_param import DataParam
from outfit_datasets.datum import getDatum

LOGGER = logging.getLogger(__name__)


class OutfitData(object):
    r"""Outfit data class. The class has the following attributes:

    - ``dataset``: :class:`torch.utils.data.Dataset`
    - ``dataloader``: :class:`torch.utils.data.DataLoader`
    - ``num_batch``: number of batches.
    - ``num_sample``: number of samples.

    """

    def __init__(self, param: DataParam = None, **kwargs):
        param = DataParam() if param is None else param
        param = attr.evolve(param, **kwargs)
        LOGGER.info(
            "Loading data (%s) in phase (%s)", torchutils.colour(param.data_set), torchutils.colour(param.phase)
        )
        LOGGER.info(
            "DataLoader configuration: batch size (%s), number of workers (%s)",
            torchutils.colour(param.batch_size),
            torchutils.colour(param.num_workers),
        )
        self.num_users = param.num_users
        self.posi_data = np.array(pd.read_csv(param.posi_fn, dtype=np.int, header=None))
        print(self.posi_data.shape)
        self.nega_data = np.array(pd.read_csv(param.nega_fn, dtype=np.int, header=None))
        print(self.nega_data.shape)
        if param.builder.data_mode == "FITB":
            data = np.array(pd.read_csv(param.fitb_fn, dtype=np.int))
            self.posi_data = data
            self.nega_data = None
            self.num_comparisons = len(data)
            LOGGER.info("Summary for fill-in-the-blank data set")
            LOGGER.info("Number of questions: %s", torchutils.colour(self.num_comparisons))
            LOGGER.info("Number of answers: %s", torchutils.colour(self.param.num_choices))

        self.datum = getDatum(param)
        self.dataset = getBuilder(
            datum=self.datum, posi_data=self.posi_data, nega_data=self.nega_data, **param.builder.asdict()
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
