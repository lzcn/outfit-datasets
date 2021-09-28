import os
from typing import List

import attr
from torchutils.param import DataReaderParam, Param


@attr.s
class BuilderParam(Param):
    data_mode = attr.ib(default="PairWise")
    pos_mode = attr.ib(default="Fix")
    neg_mode = attr.ib(default="RandomMix")
    pos_param = attr.ib(factory=dict)
    neg_param = attr.ib(factory=dict)


@attr.s
class OutfitDataParam(Param):
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
                "builder": {
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
    #: data root
    data_root: str = attr.ib(default="processed", repr=False)
    #: readers :class:`DataReaderParam`
    readers: List[DataReaderParam] = attr.ib(factory=list)
    #: builder :class:`BuilderParam`
    builder: BuilderParam = attr.ib(factory=dict, converter=BuilderParam.from_dict)
    #: number of item categories
    num_types: int = attr.ib(default=10)
    #: max size of an outfit
    max_size: int = attr.ib(default=10)
    #: number of users
    num_users: int = attr.ib(default=1, converter=int)
    # for data-lodaer
    #: batch size for dataloader
    batch_size: int = attr.ib(default=64, converter=int)
    #: number of workers for dataloader
    num_workers: int = attr.ib(default=8, converter=int)
    #: whether to shuffle the dataset
    shuffle: bool = attr.ib(default=None)
    # non-configurable attributes
    pos_fn: str = attr.ib(init=False)
    neg_fn: str = attr.ib(init=False)
    fitb_fn: str = attr.ib(init=False)

    def __attrs_post_init__(self):
        # reader parameters
        self.readers = [dict()] if len(self.readers) == 0 else self.readers
        self.readers = [DataReaderParam.from_dict(param) for param in self.readers]
        self.pos_fn = os.path.join(self.data_root, f"{self.phase}_pos")
        self.neg_fn = os.path.join(self.data_root, f"{self.phase}_neg")
        self.fitb_fn = os.path.join(self.data_root, f"{self.phase}_fitb")
        self.item_list_fn = os.path.join(self.data_root, "items.json")
