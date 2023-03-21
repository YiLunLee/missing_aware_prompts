
# Data directory structure:
Please organize the datasets as follows, otherwise you may need to revise the `write_*.py` files to meet your dataset path and files.

## MM-IMDb
[MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb) [(archive.org mirror)](https://archive.org/download/mmimdb)

    root
    ├── images            
    │   ├── 00000005.jpeg 
    │   ├── 00000008.jpeg   
    │   └── ...        
    ├── labels          
    │   ├── 00000005.json 
    │   ├── 00000008.json   
    │   └── ...        
    └── split.json 


## Food101
[UPMC Food-101](https://visiir.isir.upmc.fr/explore) [(Kaggle)](https://www.kaggle.com/datasets/gianmarco96/upmcfood101?select=texts)

    root
    ├── images            
    │   ├── train                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    │   ├── test                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    ├── texts          
    │   ├── train_titles.csv            
    │   └── test_titles.csv         
    ├── class_idx.json         
    ├── text.json         
    └── split.json


##  Hateful Memes
[Hateful Memes](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/)[(Kaggle)]](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset)

    root
    ├── img           
    │   ├── xxxxx.png 
    │   ├── xxxxx.png   
    │   └── ...        
    ├── train.jsonl          
    ├── dev.jsonl           
    └── test.jsonl 