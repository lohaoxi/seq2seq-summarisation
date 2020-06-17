# Purpose: create train_iter, valid_iter, test_iter and vocab
from data import load_tfds, save_object
import time

start = time.time()

ds_name = 'cnn_dailymail'
SIZES = (-1, -1, -1) # train_size, valid_size, test_size

param = [ds_name, SIZES]
param_name = ['ds_name', 'SIZES']
for i in range(len(param)):
    print(param_name[i], '=', param[i])
    
# Load and clean tfds.dataset
print('Loading {} tensorflow dataset...'.format(ds_name))
train_ds, valid_ds, test_ds = load_tfds(ds_name, sizes=SIZES)

print('Number of training examples: {}'.format(len(train_ds[0])))
print('Number of validation examples: {}'.format(len(valid_ds[0])))
print('Number of testing examples: {}'.format(len(test_ds[0])))

save_object(train_ds, ds_name + '_preprocessed' + '_train_ds')
save_object(valid_ds, ds_name + '_preprocessed' + '_valid_ds')
save_object(test_ds, ds_name + '_preprocessed' + '_test_ds')

print('Preprocessing complete')

end = time.time()
elapsed = end - start
print('Time elapsed for preprocessing: {}'.format(elapsed))
