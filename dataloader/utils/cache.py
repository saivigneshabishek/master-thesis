import torch
import numpy as np
import ctypes
import multiprocessing as mp

class SharedCache:
    '''made into a class following: https://github.com/ptrblck/pytorch_misc/blob/master/shared_array.py'''
    def __init__(self, dims):
        num, l, n, d = dims
        shared_array_base = mp.Array(ctypes.c_float, num*l*n*d)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(num, l, n, d)
        self.shared_array = torch.from_numpy(shared_array)

    def _store(self, idx, value):
        '''stores the value at given idx of sharedarray'''
        self.shared_array[idx] = value
    
    def _get(self, idx):
        '''gets the item stored at given idx of sharedarray'''
        return self.shared_array[idx]
    
    def _getaug(self, idx):
        item = self.shared_array[idx]
        aug = np.random.normal([0,0,0,0,0,0,0,0,0], [0.12, 0.12, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 9)
        return (item + aug)