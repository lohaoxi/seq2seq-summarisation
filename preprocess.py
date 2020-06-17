# Purpose: create train_iter, valid_iter, test_iter and vocab
import torch
from data import load_tfds, clean_df
from utils import Dataset
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds_name = 'cnn_dailymail'


TRAIN_SIZE = -1
VALID_SIZE = -1
TEST_SIZE = -1
VOCAB_SIZE = None
VOCAB_MIN_FREQ = 5
BATCH_SIZE = 4


if __name__ == '__main__':
    
    # Define saving name:
    train_iter_dir = ds_name + '_train_iter'
    valid_iter_dir = ds_name + '_valid_iter'
    test_iter_dir = ds_name + '_test_iter'
    all_iter_dir = [train_iter_dir, valid_iter_dir, test_iter_dir]
    vocab_dir = ds_name + '_vocab'
    
    # Load tfds.dataset and convert to pd.DataFrame
    sizes = (TRAIN_SIZE, VALID_SIZE, TEST_SIZE)
    print('Loading {} tensorflow dataset...'.format(ds_name))
    (train_df, valid_df, test_df) = load_tfds(ds_name, sizes)
    print('Number of training examples: {}'.format(train_df.shape[0]))
    print('Number of validation examples: {}'.format(valid_df.shape[0]))
    print('Number of testing examples: {}'.format(test_df.shape[0]))
    all_df = [train_df, valid_df, test_df]

    
    # Data Cleaning
    print('Cleaning dataframe')
    for df in all_df:
        df = clean_df(df)

    # Prepare custom dataset
    all_ds = [0] * len(all_df)
    for i in range(len(all_ds)):
        # Convert pd.DataFrame to custom dataset
        print('Convert to custom dataset {}...'.format(all_iter_dir[i]))
        ds = Dataset(all_df[i])
        all_ds[i] = ds
    train_ds, valid_ds, test_ds = all_ds
    
    # Prepare Vocab 
    print('Building vocab')
    vocab = train_ds.build_vocab(vocab_size=VOCAB_SIZE, min_freq=VOCAB_MIN_FREQ)
        
    # Prepare data iterator and generator
    all_iter = [0] * len(all_ds)
    for i in range(len(all_ds)):
        # Build data iterator for custom dataset
        print('Building data iterator for {} ...'.format(all_iter_dir[i]))
        all_iter[i] = all_ds[i].build_iterator(vocab=vocab, batch_size=BATCH_SIZE)
        
    # Save object as pickle
    preprocessed_data = {
        'preprocess_param': dict(zip(('tfds_name', 
                                      'train_size', 
                                      'valid_size', 
                                      'test_size', 
                                      'vocab_size', 
                                      'vocab_min_freq', 
                                      'batch_size'), 
                                     (ds_name, 
                                      TRAIN_SIZE, 
                                      VALID_SIZE, 
                                      TEST_SIZE, 
                                      VOCAB_SIZE, 
                                      VOCAB_MIN_FREQ, 
                                      BATCH_SIZE))),
        'vocab': vocab,
        'train_iterator': all_iter[0],
        'valid_iterator': all_iter[1],
        'test_iterator': all_iter[2],
    }
    print('Saving preprocessed_data...')
    save_path = open("preprocessed_data.pickle", "wb")
    pickle.dump(preprocessed_data, save_path)
    save_path.close()
    
    print('Preprocessing complete')
