# -*- coding: utf-8 -*-

import time
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
# from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.metrics import f1_score

# Import my modules
from mysvm import ersvm, ersvmh, enusvm, rampsvm, svmutil

# Load MNIST data
mnist = fetch_mldata('MNIST original', data_home='data/sklearn')
mnist1 = mnist.data[mnist.target == 1]  # size = 7877
mnist7 = mnist.data[mnist.target == 7]  # size = 7293
mnist9 = mnist.data[mnist.target == 9]  # size = 6958
x = np.vstack([mnist1, mnist7]).astype(float)
y = np.array([1] * len(mnist1) + [-1] * len(mnist7))
num, dim = x.shape
