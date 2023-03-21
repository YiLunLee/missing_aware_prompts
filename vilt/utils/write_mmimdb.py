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
    GENRE_CLASS = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror'
     , 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music',
     'Animation', 'Musical', 'Western', 'Sport', 'Short', 'Film-Noir']
    GENRE_CLASS_DICT = {}
    for idx, genre in enumerate(GENRE_CLASS):
        GENRE_CLASS_DICT[genre] = idx    

    image_root = os.path.join(root, 'images')
    label_root = os.path.join(root, 'labels')
    
    with open(f"{root}/split.json", "r") as fp:
        split_sets = json.load(fp)
        
    
    total_genres = []
    for split, samples in split_sets.items():
        data_list = []
        for sample in tqdm(samples):
            image_path = os.path.join(image_root, sample+'.jpeg')
            label_path = os.path.join(label_root, sample+'.json')
            with open(image_path, "rb") as fp:
                binary = fp.read()
            with open(label_path, "r") as fp:
                labels = json.load(fp)    
            
            # There could be more than one plot for a movie,
            # if single plot, only the first plots are used
            if single_plot:
                plots = [labels['plot'][0]]
            else:
                plots = labels['plot']
                
            genres = labels['genres']
            label = [1 if g in genres else 0 for g in GENRE_CLASS_DICT]
            data = (binary, plots, label, genres, sample, split)
            data_list.append(data)

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "plots",
                "label",
                "genres",
                "image_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/mmimdb_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)        