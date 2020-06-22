import pickle
import json
import h5py
from tqdm import tqdm, trange
from time import sleep

def save_object(obj, obj_name):
    save_path = open(obj_name + '.pickle', 'wb')
    print('Saving {} into pickle...'.format(obj_name))
    pickle.dump(obj, save_path)
    save_path.close()
    print('Saved ' + obj_name + '.pickle')
    
def load_object(obj_name):
    load_path = open(obj_name + '.pickle', 'rb')
    print('Loading {} from pickle...'.format(obj_name))
    obj = pickle.load(load_path)
    print('Loaded ' + obj_name + '.pickle')
    return obj

def save_json(obj, obj_name):
    with open(obj_name + '.json', 'w') as fp:
        print('Saving '.format(obj_name))
        json.dump(obj, fp)
    print('Saved {}.json'.format(obj_name))

def load_json(obj_name):
    with open(obj_name + '.json', 'r') as fp:
        print('Loading {} from json...'.format(obj_name))
        obj = json.load(fp)
    print('Loaded ' + obj_name + '.json')
    return obj

def save_hdf5(ds_name, obj_name, ds):
    with h5py.File(ds_name + '_' + obj_name + '.hdf5', 'w-') as f:
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset('src', data=ds['src'], dtype=string_dt)
        f.create_dataset('trg', data=ds['trg'], dtype=string_dt)

def save_ds(ds, path, size):
    t = trange(size, desc='Saved', leave=True)
    for i in t:
        
        idx = '_' + (len(str(size)) - len(str(i)))*'0' + str(i)
        with open(path + idx + '.json', 'w') as fp:
            json.dump(ds[i], fp)
            t.set_description('Saved file %s' % path[:len(path)//2] + idx + '.json')
            t.refresh()



def save_model():
    # Later
    pass
