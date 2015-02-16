# -*- coding: utf-8 -*-

# Import libraries
import numpy as np
import scipy as sp
import time
from sklearn.metrics import pairwise_kernels

def kernel_matrix(x, kernel):
    if kernel == 'linear':
        return np.dot(x, x.T)
    elif kernel == 'rbf':
        num, dim = x.shape
        tmp = np.dot(np.ones([num, 1]), np.array([np.linalg.norm(x, axis=1)])**2)
        return np.exp(-(tmp - 2 * np.dot(x, x.T) + tmp.T))

if __name__ == '__main__':
    x = np.array([[1,2], [3,4], [5,6]])
    x = np.ones([10000, 10])
    kernel = 'rbf'
    t1 = time.time()
    print kernel_matrix(x, kernel)
    t2 = time.time()
    print pairwise_kernels(x, metric='rbf', gamma=1.)
    t3 = time.time()
    print 'ORIGINAL:', t2-t1
    print 'scikit-learn:', t3-t2
