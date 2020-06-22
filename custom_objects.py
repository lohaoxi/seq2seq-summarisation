from collections import Counter
from torch.utils.data import IterableDataset
from itertools import cycle
import json
import numpy as np

class Vocab:
    # https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks-python.html
    def __init__(self):
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.UNK = 3
        self.init_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

        self.word2index = {}
        self.word2count = Counter()
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
        
class JSONDataset(IterableDataset):
    # https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    def __init__(self, ds_name, split, size):
        self.file_path = ds_name + '_' + str(split) + '_'
        self.size = size
        
    def parse_file(self):
        for i in range(self.size):
            with open(self.file_path[:-1] + '/' + self.file_path + (len(str(self.size)) - len(str(i)))*'0' + str(i) + '.json', 'r') as fp:
                example = json.load(fp)
            yield (np.array(example['src']), np.array(len(example['src']))), (np.array(example['trg']), np.array(len(example['trg']))) # Return ((src, src_len), (trg, trg_len))
        
    def get_stream(self):
        return cycle(self.parse_file())

    def __iter__(self):
        return self.get_stream()

    
