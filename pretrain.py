import time
from data import load_object, save_object
from builder import build_vocab, build_iter

start = time.time()
ds_name = 'cnn_dailymail'
SIZES = (1000, 1000, 1000) # train_size, valid_size, test_size

MAX_VOCAB_SIZE = 30000
VOCAB_MIN_FREQ = 5
BATCH_SIZE = 8
SORT_ITER = True
TORCH_TENSOR = True

param = [ds_name, SIZES, MAX_VOCAB_SIZE, VOCAB_MIN_FREQ, BATCH_SIZE, SORT_ITER, TORCH_TENSOR]
param_name = ['ds_name', 'SIZES', 'MAX_VOCAB_SIZE', 'VOCAB_MIN_FREQ', 'BATCH_SIZE', 'SORT_ITER', 'TORCH_TENSOR']
for i in range(len(param)):
    print(param_name[i], '=', param[i])
    
# Load preprocessed dataset from pickle
train_ds = load_object(ds_name + '_preprocessed' + '_train_ds')
valid_ds = load_object(ds_name + '_preprocessed' + '_valid_ds')
test_ds = load_object(ds_name + '_preprocessed' + '_test_ds')

# Build vocab
print('\nBuilding vocab')
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
save_object(preprocessed, ds_name + '_pretrained')
print('Pretraining complete')

end = time.time()
elapsed = end - start
print('Time elapsed for pretraining: {}'.format(elapsed))