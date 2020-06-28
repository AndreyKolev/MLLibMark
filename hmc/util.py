import numpy as np
import math
import json
from time import perf_counter
from urllib.request import urlretrieve
import sklearn.datasets as datasets
import tempfile
import os.path

def get_data():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/'
    train_file = 'a9a'
    test_file = train_file+'.t'
    
    if not os.path.exists(train_file):
        print('downloading')
        urlretrieve(url+train_file, train_file)
    if not os.path.exists(test_file):
        print('downloading')
        urlretrieve(url+test_file, test_file)
    
    X_train, y_train = datasets.load_svmlight_file(train_file)
    X_train = X_train.toarray().astype('float32')
    
    X_test, y_test = datasets.load_svmlight_file(test_file)
    X_test = X_test.toarray().astype('float32')
    X_test = np.hstack((X_test, np.zeros_like(X_test[:, :1])))
    
    y_train = ((y_train + 1)/2).astype('float32')
    y_test = ((y_test + 1)/2).astype('float32')
    
    return X_train, y_train, X_test, y_test