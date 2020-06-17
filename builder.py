from tqdm import tqdm
from collections import Counter
from utils import Vocab
import numpy as np
import torch

def sent2seq(words, vocab):
    # Input: sent: 1 list of str, 
    #        vocab: 1 Vocab
    # Output: seq: 1 list of int
    # Do: convert a list of words into a list of integers
    words = [vocab[1]] + words + [vocab[2]]
    seq = [int(vocab[word]) for word in words]
    return seq

def pad(seq_lst):
    # Input: seq_lst: 1 list of lists of int
    # Output: padded: 1 np.array
    # Do: aad a list of seqs into an np.array
    # Include: sent2seq
    m = len(seq_lst)
    max_len = max([len(ss) for ss in seq_lst])
    padded = np.zeros([m, max_len])
    for i in range(m):
        l = len(seq_lst[i])
        padded[i, 0:l] = np.array(seq_lst[i])
    return padded

def build_vocab(ds, max_size=30000, min_freq=2):
    
    punct = ['.', ',', '!', '?', '\'', '"']
    adding_words = 'Adding words...'
    filtering_vocab = 'Fitering vocab...'
    sorting_vocab = 'Sorting vocab...'
    vocab = Vocab()
    
    # Add words in vocab
    src_lst, trg_lst = ds
    for src, trg in tqdm(zip(src_lst, trg_lst), desc=adding_words):
        words = src + trg
        for word in words:
            if word.isalpha() or word in punct:
                vocab.add_word(word)
    vocab_dict = vocab.word2count.items()
    
    # Trim vocab
    vocab_dict = [(word, count) for (word, count) in tqdm(vocab_dict, desc=sorting_vocab)]
    vocab_dict= sorted(vocab_dict, key=lambda x: x[1], reverse=True)
    if max_size < len(vocab_dict):
        vocab_dict = vocab_dict[:max_size]
    
    vocab.word2index = {}
    vocab.word2count = Counter()
    vocab.index2word = vocab.init_tokens[:]
            
    for word, count in tqdm(vocab_dict, desc=filtering_vocab):
        if count >= min_freq: 
            vocab.word2index[word] = len(vocab.index2word)
            vocab.word2count[word] = count
            vocab.index2word.append(word)
    vocab.n_words = len(vocab.index2word)
        
    return vocab

def build_iter(ds, vocab, batch_size, sort=True, torch_tensor=False):
    
    converting_int = 'Converting words to integers'
    sorting_pairs = 'Sorting pairs...'
    making_minbatches = 'Making minibatches...'
    padding_minibatches = 'Padding minibatches...'
    
    src_lst, trg_lst = ds
    batches = []
    for src, trg in tqdm(zip(src_lst, trg_lst), desc=converting_int):
        src = sent2seq(src, vocab)
        trg = sent2seq(trg, vocab)
        batches.append((src, trg))
    if sort == True:
        print(sorting_pairs)
        batches = sorted(batches, key=lambda x: len(x[0]), reverse=True)

    batches = [batches[i:i+batch_size] for i in tqdm(range(0, len(batches), batch_size), desc=making_minbatches)]
    iterator = []
    for batch in tqdm(batches, desc=padding_minibatches):
        src_lst = []
        trg_lst = []
        for pair in batch:
            src_lst.append(pair[0])
            trg_lst.append(pair[1])
        src_arr = pad(src_lst)
        trg_arr = pad(trg_lst)
        src_arr = src_arr.T
        trg_arr = trg_arr.T
        if torch_tensor == True:
            batch = torch.from_numpy(src_arr), torch.from_numpy(trg_arr)
        else:
            batch = src_arr, trg_arr # From 1 list of tuples to 1 tuple of 2 np.array
            
        iterator.append(batch)
    
    return iterator