# Prepare data
import tensorflow_datasets as tfds
import urllib3
from tqdm import tqdm
from collections import Counter
from cnn_dailymail_clean import clean_src, clean_trg
from save_load import save_json, save_ds
from custom_objects import Vocab
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Tools

def preprocess(ds):
    # Input: tfds
    # Clean sentences
    # Tokenize sentences
    # Return a preporcessed dataset (dict)
    srcs = []
    trgs = []
    for example in tqdm(tfds.as_numpy(ds)):
        
        src = example['article']
        trg = example['highlights']
        
        src = clean_src(src)
        trg = clean_trg(trg)

        srcs.append(src.split())
        trgs.append(trg.split())
        
    return {'src': srcs, 'trg': trgs}

def build_vocab(ds, max_size=80000, min_freq=2):
    # Input: preprocessed dataset (dict)
    # Return Vocab
    print('Building vocab...')
    vocab = Vocab()

   
    # Add words in vocab
    src_lst = ds['src']
    trg_lst = ds['trg']
    for src, trg in tqdm(zip(src_lst, trg_lst), desc='Adding words...'):
        if type(src) == str:
            src = src.split()
        if type(trg) == str:
            trg = trg.split()
        words = src + trg
        for word in words:
            if word.isalpha() or word in ['.', ',', '!', '?', '\'', '"']:
                vocab.add_word(word)
    vocab_count = list(vocab.word2count.items())
    
    # Trim size
    vocab_count = sorted(vocab_count, key=lambda x: x[1], reverse=True) # Sort by counts (descending)
    if max_size < len(vocab_count):
        vocab_count = vocab_count[:max_size-4]
    
    # Filter less frequent words
    vocab.word2index = {}
    vocab.word2count = Counter()
    vocab.index2word = vocab.init_tokens[:]
    vocab_count = sorted(vocab_count, key=lambda x: x[0]) # Sort by words (a-z)

    for word, count in vocab_count:
        if count >= min_freq: 
            vocab.word2index[word] = len(vocab.index2word)
            vocab.word2count[word] = count
            vocab.index2word.append(word)
    vocab.n_words = len(vocab.index2word)

    return vocab

def postprocess(ds_dict, vocab):
    # Input: preprocessed dataset (dict), Vocab
    # Encode words into integers
    # Add init_token and end token in each sentence
    # Defne an exampls as a dict of src and trg
    # Return a list of examples
    examples = []
    for src, trg in tqdm(zip(ds_dict['src'], 
                             ds_dict['trg'])):
        
        src = [vocab.SOS] + [vocab[word] for word in src] + [vocab.EOS]
        trg = [vocab.SOS] + [vocab[word] for word in trg] + [vocab.EOS]
        
        examples.append({'src': src, 'trg': trg})
        
    return examples

# Import data
    
ds_name = 'cnn_dailymail'
ds, info = tfds.load(ds_name, with_info=True, download=False, data_dir='E:')

train_ds = ds['train']
valid_tfds = ds['validation']
test_tfds = ds['test']

train_size = info.splits['train'].num_examples
valid_size = info.splits['validation'].num_examples
test_size = info.splits['test'].num_examples

# Preprocess data

train_ds = preprocess(train_ds)
valid_ds = preprocess(valid_tfds)
test_ds = preprocess(test_tfds)

# Create vocab from train data
VOCAB_MAX_SIZE = 50000
VOCAB_MIN_FREQ = 2
vocab = build_vocab(ds=train_ds,
                    max_size=VOCAB_MAX_SIZE,
                    min_freq=VOCAB_MIN_FREQ)
print('Done')

# Save vocab, just the vocab.word2count

vocab_name = ds_name + '_vocab' + '_' + (8-len(str(VOCAB_MAX_SIZE)))*'0' + str(VOCAB_MAX_SIZE) + '_' + (2-len(str(VOCAB_MIN_FREQ)))*'0' + str(VOCAB_MIN_FREQ)
print('Saving vocab as {}.json...'.format(vocab_name))
save_json(vocab.word2count, vocab_name)

# Postprocess data

train_ds = postprocess(train_ds, vocab)
valid_ds = postprocess(valid_ds, vocab)
test_ds = postprocess(test_ds, vocab)

# Save all examples individually

train_path = ds_name + '_train/' + ds_name + '_train'
valid_path = ds_name + '_validation/' + ds_name + '_validation'
test_path = ds_name + '_test/' + ds_name + '_test'

save_ds(train_ds, train_path, train_size)
save_ds(valid_ds, valid_path, valid_size)
save_ds(test_ds, test_path, test_size)

# Save all 

