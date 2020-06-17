import tensorflow_datasets as tfds
from tqdm import tqdm
import urllib3
from contraction import contraction_map
import re
import pickle
# from contraction import contraction_map

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
tfds.disable_progress_bar()

# =============================================================================
# from tqdm import tqdm as tqdm_base
# def tqdm(*args, **kwargs):
#     # Prevent tqdm progress bar starting on new line
#     # https://stackoverflow.com/questions/41707229/tqdm-printing-to-newline
#     if hasattr(tqdm_base, '_instances'):
#         for instance in list(tqdm_base._instances):
#             tqdm_base._decr_instances(instance)
#     return tqdm_base(*args, **kwargs)
# =============================================================================

# =============================================================================
# Tensorflow dataset loading
# =============================================================================

def tfds2tuple(ds, size, desc):
    # Convert tf.data.Dataset.to 1 tuple of 2 lists
    src_lst = []
    trg_lst = []
    for example in tqdm(tfds.as_numpy(ds), desc):
        src = example['article']
        trg = example['highlights']
        src_lst.append(src.decode("utf-8"))
        trg_lst.append(trg.decode("utf-8"))
        if len(src_lst) == len(trg_lst) == size:
            break
    return src_lst, trg_lst

def load_tfds(ds_name, sizes):
    ds, info = tfds.load(ds_name, with_info=True)
    train_ds = ds['train']
    valid_ds = ds['validation']
    test_ds = ds['test']
    train_size, valid_size, test_size = sizes
    train_ds = tfds2tuple(train_ds, train_size, desc='Re-establishing training dataset...')
    valid_ds = tfds2tuple(valid_ds, valid_size, desc='Re-establishing validation dataset...')
    test_ds = tfds2tuple(test_ds, test_size, desc='Re-establishing testing dataset...')
    return (train_ds, valid_ds, test_ds)

# =============================================================================
# Data Saving
# =============================================================================
    
def save_object(obj, obj_name):
    # Input: obj: object
    #        obj_name: str
    # Save an objet into a pickle
    save_path = open(obj_name + '.pickle', 'wb')
    print('Saving {} into pickle...'.format(obj_name))
    pickle.dump(obj, save_path)
    save_path.close()
    print('Saved {} into pickle'.format(obj_name))
    
def load_object(obj_name):
    load_path = open(obj_name + '.pickle', 'rb')
    print('Loading {} from pickle...'.format(obj_name))
    obj = pickle.load(load_path)
    print('Loaded {} from pickle'.format(obj_name))
    return obj

def save_model():
    # Later
    pass

# =============================================================================
# Data Cleaning
# =============================================================================

def clean_cnn(sentence):
    # Strip (CNN) -- and (CNN)  -- if it exists then strip
    headers = ['(CNN) -- ', '(CNN)  --']
    for header in headers:
        ind = sentence.find(header)
        if ind > -1:
            sentence = sentence[ind+len(header):]
    return sentence.strip()

def clean_author(sentence):
    # eg. Strip By . Ellie Zolfagharifard . if it exist then strip
    author = re.search("By . [a-zA-Z\s]+", sentence)
    if author != None:
        ind = author.end()
        sentence = sentence[ind:]
    return sentence.strip()

def clean_publish(sentence):
    # eg. Strip PUBLISHED: . 03:39 EST, 8 May 2012 . if it exist then strip
    publish = re.search("PUBLISHED: . \d\d:\d\d\s[A-Z]{3},\s\d{1,2}\s[a-zA-Z]+\s\d{4}", sentence)
    if publish != None:
        ind = publish.end()
        sentence = sentence[ind:]
    return sentence.strip()

def clean_update(sentence):
    # eg. Strip UPDATED: . 18:10 EST, 8 May 2012 . if it exist then strip
    update = re.search("UPDATED: . \d\d:\d\d\s[A-Z]{3},\s\d{1,2}\s[a-zA-Z]+\s\d{4}", sentence)
    if update != None:
        ind = update.end()
        sentence = sentence[ind:]
    return sentence.strip()

def clean_punct(sentence):
    # Remove \n \t
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace('\t', '')
    # Remove |
    sentence = sentence.replace('|', '')
    # Remove \
    sentence = sentence.replace('\\', '')
    # Remove . if it is the first word
    ind = sentence.find('.')
    if ind == 0:
        sentence = sentence[1:]
    return sentence.strip()

def clean_bracket(sentence):
    sentence = re.sub("\([(\d|\D)]+\)", '', sentence)
    return sentence
    
def clean_src(lst):
    clean = []
    for sentence in tqdm(lst):
        sentence = contraction_map(sentence)
        sentence = clean_cnn(sentence)
        sentence = clean_author(sentence)
        sentence = clean_publish(sentence)
        sentence = clean_update(sentence)
        sentence = clean_bracket(sentence)
        sentence = clean_punct(sentence)
        sentence = sentence.lower()
        clean.append(sentence.split())
    return clean

def clean_trg(lst):
    clean = []
    for sentence in tqdm(lst):
        sentence = contraction_map(sentence)
        sentence = clean_bracket(sentence)
        sentence = clean_punct(sentence)
        sentence = sentence.lower()
        clean.append(sentence.split())
    return clean

def clean_ds(ds):
    src_lst, trg_lst = ds
    src_lst = clean_src(src_lst)
    trg_lst = clean_trg(trg_lst)
    return src_lst, trg_lst

