from vilt.datasets import FOOD101Dataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class FOOD101DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return FOOD101Dataset

    @property
    def dataset_name(self):
        return "food101"

    def setup(self, stage):
        super().setup(stage)

