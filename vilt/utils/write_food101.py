import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word

def make_arrow(root, dataset_root, single_plot=False, missing_type=None):
    image_root = os.path.join(root, 'images')
    
    with open(f"{root}/class_idx.json", "r") as fp:
        FOOD_CLASS_DICT = json.load(fp)
        
    with open(f"{root}/text.json", "r") as fp:
        text_dir = json.load(fp)
        
    with open(f"{root}/split.json", "r") as fp:
        split_sets = json.load(fp)
        
    
    for split, samples in split_sets.items():
        split_type = 'train' if split != 'test' else 'test'
        data_list = []
        for sample in tqdm(samples):
            if sample not in text_dir:
                print("ignore no text data: ", sample)
                continue
            cls = sample[:sample.rindex('_')]
            label = FOOD_CLASS_DICT[cls]
            image_path = os.path.join(image_root, split_type, cls, sample)

            with open(image_path, "rb") as fp:
                binary = fp.read()
                
            text = [text_dir[sample]]
            
            
            data = (binary, text, label, sample, split)
            data_list.append(data)

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "text",
                "label",
                "image_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/food101_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)        