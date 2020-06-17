from collections import Counter

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
        
    def stat(self):
        # Later
        pass
        
    
# =============================================================================
# def calculate_batch_pad_prop(batch):
#     # Input: batch: Batch
#     # Ouput (src_pad_prop, trg_pad_prop): 1 tuple of 2 int
#     # Calculate the proportion of PAD (0) in the batch
#     src_arr = batch.src.numpy().T
#     trg_arr = batch.trg.numpy().T
#     src_pad_prop = (src_arr.size - np.count_nonzero(src_arr)) / src_arr.size
#     trg_pad_prop = (trg_arr.size - np.count_nonzero(trg_arr)) / trg_arr.size
#     return (src_pad_prop, trg_pad_prop)
# 
# def average_pad_prop(iterator_lst):
#     # Input: 1 iterator_lst
#     # Output: 1 tuple of 2 int
#     # Do: Calulate the average proportion of padding in the list of batches
#     # Include: calculate_batch_pad_prop
#     src_pad_props = []
#     trg_pad_props = []
#     for batch in iterator_lst:
#         (src_pad_prop, trg_pad_prop) = calculate_batch_pad_prop(batch)
#         src_pad_props.append(src_pad_prop)
#         trg_pad_props.append(trg_pad_prop)
#     return (np.mean(src_pad_props), np.mean(trg_pad_props))
# 
# def calculate_batch_unk_prop(batch):
#     # Input: batch; Batch
#     # Ouput (src_unk_prop, trg_unk_prop): 1 tuple of 2 int
#     # Calculate the proportion of UNK (3) in the batch
#     src_arr = batch.src.numpy().T
#     trg_arr = batch.trg.numpy().T
#     src_unk_prop = np.count_nonzero(src_arr == 3) / src_arr.size
#     trg_unk_prop = np.count_nonzero(trg_arr == 3) / trg_arr.size
#     return (src_unk_prop, trg_unk_prop)
# 
# def average_unk_prop(iterator_lst):
#     # Input: 1 iterator_lst
#     # Output: 1 tuple of 2 int
#     # Do: Calulate the average proportion of unk in the list of batches
#     # Include: calculate_batch_unk_prop
#     src_unk_props = []
#     trg_unk_props = []
#     for batch in iterator_lst:
#         (src_unk_prop, trg_unk_prop) = calculate_batch_unk_prop(batch)
#         src_unk_props.append(src_unk_prop)
#         trg_unk_props.append(trg_unk_prop)
#     return (np.mean(src_unk_props), np.mean(trg_unk_props))
#     
# =============================================================================




