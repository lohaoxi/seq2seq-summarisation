# Purpose: create train_iter, valid_iter, test_iter and vocab
from data import load_tfds, clean_ds, save_object
from builder import build_vocab, build_iter
import time

start = time.time()
print('\n')

ds_name = 'cnn_dailymail'
SIZES = (-1, -1, -1) # train_size, valid_size, test_size
MAX_VOCAB_SIZE = 30000
VOCAB_MIN_FREQ = 5
BATCH_SIZE = 8
SORT_ITER = True
TORCH_TENSOR = True

param = [ds_name, SIZES, MAX_VOCAB_SIZE, VOCAB_MIN_FREQ, BATCH_SIZE, SORT_ITER, TORCH_TENSOR]
param_name = ['ds_name', 'SIZES', 'MAX_VOCAB_SIZE', 'VOCAB_MIN_FREQ', 'BATCH_SIZE', 'SORT_ITER', 'TORCH_TENSOR']
for i in range(len(param)):
    print(param_name[i], ' = ', param[i])
    
# Load tfds.dataset
print('Loading {} tensorflow dataset...'.format(ds_name))
train_ds, valid_ds, test_ds = load_tfds(ds_name, sizes=SIZES)
print('Number of training examples: {}'.format(len(train_ds[0])))
print('Number of validation examples: {}'.format(len(valid_ds[0])))
print('Number of testing examples: {}'.format(len(test_ds[0])))
all_ds = [train_ds, valid_ds, test_ds]

# Data Cleaning
print('\nCleaning dataset')
all_ds = [clean_ds(ds) for ds in all_ds]
print('Cleaned dataset')

# Prepare Vocab 
print('\nBuilding vocab')
train_ds, valid_ds, test_ds = all_ds
vocab = build_vocab(train_ds, max_size=MAX_VOCAB_SIZE, min_freq=VOCAB_MIN_FREQ)
print('Built vocab, vocab_size = {}'.format(vocab.n_words - 4))

# Prepare Iterator
print('\nBuilding train iterator')
train_iter = build_iter(train_ds, vocab, batch_size=BATCH_SIZE, sort=SORT_ITER, torch_tensor=TORCH_TENSOR)
print('Built train iterators, batch size = {}, number of batches: {}'.format(BATCH_SIZE, len(train_iter)))

print('\nBuilding valid iterator')
valid_iter = build_iter(valid_ds, vocab, batch_size=BATCH_SIZE, sort=SORT_ITER, torch_tensor=TORCH_TENSOR)
print('Built valid iterators, batch size = {}, number of batches: {}'.format(BATCH_SIZE, len(valid_iter)))

print('\nBuilding test iterator')
test_iter = build_iter(test_ds, vocab, batch_size=BATCH_SIZE, sort=SORT_ITER, torch_tensor=TORCH_TENSOR)
print('Built test iterators, batch size = {}, number of batches: {}'.format(BATCH_SIZE, len(test_iter)))

print('Built all 3 iterators')


# Save object as pickle
print('\n')
preprocessed = {
        'param': param,
        'vocab': vocab,
        'train_iter': train_iter,
        'valid_iter': valid_iter,
        'test_iter': test_iter
        }
save_object(preprocessed, ds_name + '_preprocessed')

end = time.time()
elapsed = end - start
print('Time elapsed: {}'.format(elapsed))
