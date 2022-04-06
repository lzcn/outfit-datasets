import logging

from torch.utils.data import DataLoader
from torchutils import colour

from outfit_datasets.dataset import getOutfitData
from outfit_datasets.datum import getDatum
from outfit_datasets.param import OutfitLoaderParam

from . import utils

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
        self.pos_data = utils.load_outfit_tuples(param.pos_fn, param.phase)
        self.neg_data = utils.load_outfit_tuples(param.neg_fn, param.phase)
        if param.dataset.data_mode == "FITB":
            self.param.shuffle = False
            if param.dataset.neg_mode == "Fix":
                # use pre-defiend FITB tuples
                tuples = utils.load_outfit_tuples(param.fitb_fn, f"{param.phae}-fitb")
                assert tuples is not None, "Must provide {}_fitb for FITB task".format(param.phase)
                num_answers = param.num_fitb_choices
                num_questions = len(tuples) // num_answers
                num_cols = tuples.shape[-1]
                fitb_data = tuples.reshape((num_questions, num_answers, -1))
                pos_data, neg_data = fitb_data[:, 1, :], fitb_data[:, 1:, :]
                self.pos_data = pos_data.reshape((-1, num_cols))
                self.neg_data = neg_data.reshape((-1, num_cols))
            else:
                # generate tuples for each run
                num_answers = param.dataset.neg_param.get("ratio", 1) + 1
                num_questions = len(self.pos_data)
            LOGGER.info("Summary for fill-in-the-blank data set")
            LOGGER.info("Number of FITB questions: %s", colour(f"{num_questions:,}"))
            LOGGER.info("Number of FITB answers: %s", colour(num_answers))
            self.num_fitb_choices = num_answers
        else:
            self.num_fitb_choices = param.num_fitb_choices
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
