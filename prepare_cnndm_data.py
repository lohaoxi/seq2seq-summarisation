import tensorflow_datasets as tfds
import pandas as pd
from tqdm import tqdm
from cnndm_clean import clean_src, clean_trg

data_dir = 'E:/'
ds = tfds.load(name='cnn_dailymail', download=False, data_dir=data_dir)
train_ds = ds['train']
valid_ds = ds['validation']
test_ds = ds['test']

# Clean dataset

srcs = []
trgs = []
for ds in [train_ds, valid_ds, test_ds]:
    for example in tqdm(tfds.as_numpy(ds)):
            
        src = example['article']
        trg = example['highlights']
            
        src = clean_src(src)
        trg = clean_trg(trg)
            
        srcs.append(src)
        trgs.append(trg)
        
ds = {'src': srcs, 'trg': trgs}

# Filter dataset

max_src_len = 300
min_src_len = 50
max_trg_len = 80
min_trg_len = 10

srcs = []
trgs = []
for src, trg in tqdm(zip(ds['src'], ds['trg'])):
    if min_src_len < len(src.split()) < max_src_len and min_trg_len < len(trg.split()) < max_trg_len:
        srcs.append(src)
        trgs.append(trg)
        
filtered = {'src': srcs, 'trg': trgs}

pd.DataFrame(filtered).to_csv('cnndm.csv')
