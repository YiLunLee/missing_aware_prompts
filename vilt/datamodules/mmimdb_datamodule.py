from vilt.datasets import MMIMDBDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class MMIMDBDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MMIMDBDataset

    @property
    def dataset_name(self):
        return "mmimdb"

    def setup(self, stage):
        super().setup(stage)

