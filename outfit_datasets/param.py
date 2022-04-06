import logging
import os
from typing import List, Union

import attr
from torchutils.param import DataReaderParam, OptimParam, Param, to_param

LOGGER = logging.getLogger(__name__)


@attr.s
class OutfitDataParam(Param):
    #: dataset format
    data_mode = attr.ib(default="PairwiseOutfit")
    #: positive tuples generation mode
    pos_mode = attr.ib(default="Fix")
    #: negative tuples generation mode
    neg_mode = attr.ib(default="RandomMix")
    #: configuration for positive tuples generation
    pos_param = attr.ib(factory=dict)
    #: configuration for negative tuples generation
    neg_param = attr.ib(factory=dict)


@attr.s
class OutfitLoaderParam(Param):
    """Configuration class for data loader.

    Examples:

        .. code-block:: python

            kwargs = {
                "num_users": 1,
                "max_size": 8,
                "num_types": 8,
                "data_root": "data/maryland-polyvore",
                "readers": [
                    {
                        "reader": "TensorLMDB",
                        "path": "data/maryland-polyvore/features/resnet34",
                    },
                ],
                "dataset": {
                    "data_mode": "PointWise",
                    "posi_mode": "Fix",
                    "nega_mode": "RandomMix",
                },
                "batch_size": 64,
                "shuffle": False,
                "num_workers": 4,
                "phase": "train",
            }

            data_param = DataParam(**kwargs)
            outfit_data = OutfitData(data_param)
            for batch in outfit_data:
                // ....
    """

    # basic configurations
    #: data split of ["train", "valid", "test"]
    phase: str = attr.ib(default="train")
    #: data root, where tuples files are saved
    data_root: str = attr.ib(default="polyvore-u/processed/original/tuples_32", repr=False)
    #: readers :class:`DataReaderParam`
    readers: List[DataReaderParam] = attr.ib(factory=list)
    #: dataset :class:`OutfitDataParam`
    dataset: OutfitDataParam = attr.ib(factory=dict, converter=OutfitDataParam.from_dict)
    #: number of item categories
    num_types: int = attr.ib(default=10)
    #: max number of items in an outfit
    max_items: int = attr.ib(default=None)
    #: number of users
    num_users: int = attr.ib(default=1, converter=int)
    #: number of choice for FITB task
    num_fitb_choices: int = attr.ib(default=4, converter=int)
    # for data-lodaer
    #: batch size for dataloader
    batch_size: int = attr.ib(default=64, converter=int)
    #: number of workers for dataloader
    num_workers: int = attr.ib(default=8, converter=int)
    drop_last: bool = attr.ib(default=False)
    #: whether to shuffle the dataset
    shuffle: bool = attr.ib(default=None)
    # non-configurable attributes
    pos_fn: str = attr.ib(init=False)
    neg_fn: str = attr.ib(init=False)
    # pos_fitb_fn: str = attr.ib(init=False, repr=False)
    # neg_fitb_fn: str = attr.ib(init=False, repr=False)
    # fitb_fn: str = attr.ib(init=False, repr=False)
    # retrieval_fn: str = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):
        data_root = os.path.expanduser(self.data_root)
        # reader parameters
        self.readers = [dict()] if len(self.readers) == 0 else self.readers
        self.readers = [DataReaderParam.from_dict(param) for param in self.readers]
        self.pos_fn = os.path.join(data_root, f"{self.phase}_pos")
        self.neg_fn = os.path.join(data_root, f"{self.phase}_neg")
        self.pos_fitb_fn = os.path.join(data_root, f"{self.phase}_pos_fitb")
        self.neg_fitb_fn = os.path.join(data_root, f"{self.phase}_neg_fitb")
        self.fitb_fn = os.path.join(data_root, f"{self.phase}_fitb")
        self.retrieval_fn = os.path.join(data_root, "retrieval")
        self.item_list_fn = os.path.join(data_root, "items.json")


@attr.s
class RunParam(Param):
    r"""Configuration interface for training/testing."""

    epochs: int = attr.ib(default=100)
    data_param: Param = attr.ib(factory=dict, converter=to_param)
    valid_data_param: Param = attr.ib(factory=dict)
    test_data_param: Param = attr.ib(factory=dict)
    train_data_param: Param = attr.ib(factory=dict)
    net_param: Param = attr.ib(factory=dict, converter=to_param)
    optim_param: OptimParam = attr.ib(factory=dict)
    summary_interval: int = attr.ib(default=10)
    display_interval: int = attr.ib(default=50)
    load_trained: str = attr.ib(default=None)
    log_dir: str = attr.ib(default="summaries/log")
    log_level: str = attr.ib(default="INFO")
    gpus: Union[int, list] = attr.ib(default=0)
    num_runs: int = attr.ib(default=1)

    def __attrs_post_init__(self):
        if isinstance(self.data_param, Param):
            self.train_data_param = attr.evolve(self.data_param, **self.train_data_param)
            self.valid_data_param = attr.evolve(self.data_param, **self.valid_data_param)
            self.test_data_param = attr.evolve(self.data_param, **self.test_data_param)
        self.optim_param = OptimParam(**self.optim_param)
        # for gpus
        gpus = [self.gpus] if isinstance(self.gpus, int) else self.gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        self.gpus = list(range(len(gpus)))
