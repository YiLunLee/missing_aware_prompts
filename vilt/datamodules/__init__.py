from .mmimdb_datamodule import MMIMDBDataModule
from .hatememes_datamodule import HateMemesDataModule
from .food101_datamodule import FOOD101DataModule

_datamodules = {
    "mmimdb": MMIMDBDataModule,
    "Hatefull_Memes": HateMemesDataModule,
    "Food101": FOOD101DataModule,
}
