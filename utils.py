from collections import Counter
import numpy as np
from tqdm import tqdm 
import torch


def sent2seq(sent, vocab):
    # Input: sent: 1 list of str, 
    #        vocab: 1 Vocab
    # Output: seq: 1 list of int
    # Do: convert a list of words into a list of integers
    seq = [int(vocab[word]) for word in sent]
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

def minibatch2seq(minibatch, vocab):
    # Input: minibatch: 1 list of Examples
    #        vocab: 1 Vocab
    # Output: (src_arr, trg_arr): 1 tuple of 2 np.arrays
    # Do: convert 1 list of examples to 2 list of list of integers as a tuple
    # Include: sent2seq, pad
    assert type(vocab) == Vocab
    src_lst = []
    trg_lst = []
    for example in minibatch:
        src = sent2seq(example.src, vocab)
        trg = sent2seq(example.trg, vocab)
        src_lst.append(src)
        trg_lst.append(trg)
    src_arr = pad(src_lst)
    trg_arr = pad(trg_lst)
    return (src_arr, trg_arr)

def tokenize(sentence, reverse=False):
    # Input: sentence: 1 str
    #        reverse: 1 bool
    # Output: output: 1 list
    assert type(sentence) == str
    output = sentence.split()
    if reverse == True:
        output = output[::-1]
    return output

class Vocab:
    # Custom vocab object
    # https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks-python.html
    def __init__(self):
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.UNK = 3
        self.word2index = {}
        self.word2count = Counter()
        self.init_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.init_tokens[:]
        self.n_words = len(self.index2word)
        
    def __getitem__(self, i):
        # Input: 1 index: int / 1 word: str
        # Output 1 str / 1 int
        # Do: return the corresponding word / index
        if type(i) == int:
            return str(self.index2word[i])
        elif type(i) == str:
            return int(self.word2index.get(i, self.UNK)) # Return the index of <UNK> if input word is not in the vocab
        
    def add_word(self, word):
        # Input: word: 1 str
        # Do: add the string into vocab
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words = len(self.index2word)
        self.word2count[word] += 1
    
    def add_sent(self, sent):
        # Input: sent: 1 list of str
        # Do: add all words from the sentence into vocab
        # Include: add_word
        if type(sent) == str:
            sent = sent.split()
        for word in sent:
            self.add_word(word)
    
    def add_example(self, example):
        # Input: example: 1 Example
        # Do: add src and trg from the example into vocab
        # Include: add_sent
        self.add_sent(example.src)
        self.add_sent(example.trg)
                
    def limit(self, vocab_size, min_freq):
        # Input: vocab_size: 1 int
        #        min_freq: 1 int
        # Do: Resize the vocab according to vocab_size and min_freq
        unsorted_vocab = self.word2count.items()
        unsorted_vocab = {word: unsorted_vocab[word] for word in tqdm(self.index2word, desc='filtering words...') if unsorted_vocab[word] >= min_freq}
        sorted_vocab = sorted([(word, count) for (word, count) in tqdm(unsorted_vocab, desc='Sorting vocab')], key=lambda x: x[1], reverse=True)
        if vocab_size != None:
            assert vocab_size <= self.n_words
            sorted_vocab = sorted_vocab[:vocab_size]
            
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = self.init_tokens[:]
        
        for (word, count) in tqdm(sorted_vocab, desc='Redefining vocab...'):
            self.word2index[word] = len(self.index2word)
            self.word2count[word] = count
            self.index2word.append(word)
        self.n_words = len(self.index2word)
        
    def get_embedding():
        # Later
        pass


class Example:
    # Custom example object
    # Input: src: 1 list / 1 str 
    #        trg: 1 list / 1 str
    # Include: tokenize
    def __init__(self, src, trg):
        SOS = '<SOS>'
        EOS = '<EOS>'
        if type(src) == str:
            src = tokenize(src, reverse=True)
        if type(trg) == str:
            trg = tokenize(trg)
        self.src = [SOS] + src + [EOS]
        self.trg = [SOS] + trg + [EOS]
        self.src_len = len(self.src)
        self.trg_len = len(self.trg)
        
class Batch:
    # Custom batch object
    # Input: src_arr: 1 np.array
    #        trg_arr: 1 np.array
    #        torch_tensor: 1 bool
    def __init__(self, src_arr, trg_arr, torch_tensor=False):
        self.src = src_arr.T
        self.trg = trg_arr.T
        if torch_tensor == True:
            self.as_tensor()
        self.batch_size = self.src.shape[1]
        self.src_len = self.src.shape[0]
        self.trg_len = self.trg.shape[0]
        
    def as_tensor(self):
        self.src = torch.from_numpy(self.src)
        self.trg = torch.from_numpy(self.trg)
        
    def as_numpy(self):
        self.src = self.src.numpy()
        self.trg = self.src.numpy()

    def __getitem__(self, i):
        # Input: 1 index: int
        # Output: 1 tuple of 2 np.arrays
        # Do: return the corresponding src, trg
        return (self.src[i], self.trg[i])

class Dataset:
    # Custom dataset object
    # Input: pd.DataFrame
    #        max_src_len: int
    #        max_trg_len: int
    #        size: int
    def __init__(self, df, max_src_len=None, max_trg_len=None, size=None):
        self.df = df
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        self.size = size
        self.shape = df.shape
        self.n_examples = len(df)
        self.src = df.iloc[:, 0].tolist()
        self.trg = df.iloc[:, 1].tolist()
        self.examples = [Example(src, trg) for (src, trg) in tqdm(zip(self.src, self.trg), desc='Creating dataset of size {}...'.format(self.n_examples))] # List of examples
    

    def build_vocab(self, vocab_size=None, min_freq=1):
        # Input: vocab_size: int
        #        min_freq: int
        # Output: vocab; Vocab
        # Do: Create a Vocab object from the dataset
        vocab = Vocab()
        for example in tqdm(self.examples, desc='Vocab...'):
            vocab.add_example(example)
        vocab.limit(vocab_size, min_freq)
        self.vocab = vocab
        return vocab
        
    def build_iterator(self, vocab, batch_size=128, sort=True):
        # Input: vocab: Vocab
        #        batch_size: int
        #        sort: bool
        # Output: output: 1 list of batches
        # Do: Create a list of batches accroding to the vocab and batch_size
        if sort == True:
            sorted_examples = sorted(self.examples, key=lambda x: x.src_len, reverse=True)
            self.examples = sorted_examples
        batches = [self.examples[x:x+batch_size] for x in tqdm(range(0, self.n_examples, batch_size), desc='Slicing examples into batches...')]
        self.n_minibatches = len(batches)
        iterator_lst = []
        for mb in tqdm(batches, desc='Creating batches...'): # mb is a list of examples
            (src_arr, trg_arr) = minibatch2seq(mb, vocab)
            batch = Batch(src_arr, trg_arr, torch_tensor=False)
            iterator_lst.append(batch)
        self.iterator_list = iterator_lst
        return iterator_lst
    
    def __getitem__(self, i):
        # Input: 1 index: int
        # Output: 1 example
        # Do: return the corresponding example
        return self.examples[i]

def calculate_batch_pad_prop(batch):
    # Input: batch; Batch
    # Ouput (src_pad_prop, trg_pad_prop): 1 tuple of 2 int
    # Calculate the proportion of PAD (0) in the batch
    src_arr = batch.src.numpy().T
    trg_arr = batch.trg.numpy().T
    src_pad_prop = (src_arr.size - np.count_nonzero(src_arr)) / src_arr.size
    trg_pad_prop = (trg_arr.size - np.count_nonzero(trg_arr)) / trg_arr.size
    return (src_pad_prop, trg_pad_prop)

def average_pad_prop(iterator_lst):
    # Input: 1 iterator_lst
    # Output: 1 tuple of 2 int
    # Do: Calulate the average proportion of padding in the list of batches
    # Include: calculate_batch_pad_prop
    src_pad_props = []
    trg_pad_props = []
    for batch in iterator_lst:
        (src_pad_prop, trg_pad_prop) = calculate_batch_pad_prop(batch)
        src_pad_props.append(src_pad_prop)
        trg_pad_props.append(trg_pad_prop)
    return (np.mean(src_pad_props), np.mean(trg_pad_props))

def calculate_batch_unk_prop(batch):
    # Input: batch; Batch
    # Ouput (src_unk_prop, trg_unk_prop): 1 tuple of 2 int
    # Calculate the proportion of UNK (3) in the batch
    src_arr = batch.src.numpy().T
    trg_arr = batch.trg.numpy().T
    src_unk_prop = np.count_nonzero(src_arr == 3) / src_arr.size
    trg_unk_prop = np.count_nonzero(trg_arr == 3) / trg_arr.size
    return (src_unk_prop, trg_unk_prop)

def average_unk_prop(iterator_lst):
    # Input: 1 iterator_lst
    # Output: 1 tuple of 2 int
    # Do: Calulate the average proportion of unk in the list of batches
    # Include: calculate_batch_unk_prop
    src_unk_props = []
    trg_unk_props = []
    for batch in iterator_lst:
        (src_unk_prop, trg_unk_prop) = calculate_batch_unk_prop(batch)
        src_unk_props.append(src_unk_prop)
        trg_unk_props.append(trg_unk_prop)
    return (np.mean(src_unk_props), np.mean(trg_unk_props))
    

# =============================================================================
# train_dataset = Dataset(train_df.head(100))
# vocab = train_dataset.build_vocab(min_freq=2)
# C = train_dataset.build_iterator(vocab=vocab, batch_size=4)
# print('\n', average_pad_prop(C))
# D = (n for n in C)
# print('\n', average_unk_prop(C))
# =============================================================================



