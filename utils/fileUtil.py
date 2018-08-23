import os
import gzip
import codecs
import h5py
import pickle


def h5save(_file, _data):
    '''
    保存数据到h5py文件中
    '''
    f = h5py.File(_file, 'w')
    f['data'] = _data
    f.close()


def h5open(_file):
    '''
    读取h5py文件中的数据
    '''
    f = h5py.File(_file, 'r')
    _data = f['data'][:]
    f.close()
    return _data


def h5save_dict(_file, _data):
    '''
    保存字典数据到h5py文件中
    '''
    f = h5py.File(_file, 'w')
    for k, v in _data.items():
        f[k] = v
    f.close()


def h5open_dict(_file):
    '''
    读取h5py文件中的字典数据并返回
    '''
    _dict = {}
    f = h5py.File(_file, 'r')
    for k, v in f.items():
        _dict[k] = v[:]
    f.close()
    return _dict
