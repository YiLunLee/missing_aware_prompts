from vilt.datasets import HateMemesDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class HateMemesDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return HateMemesDataset

    @property
    def dataset_name(self):
        return "Hatefull_Memes"

    def setup(self, stage):
        super().setup(stage)
