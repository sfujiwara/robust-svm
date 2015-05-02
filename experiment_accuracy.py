import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import time
import pandas as pd

from src import ersvmdca, rampsvm, enusvm

if __name__ == '__main__':
    filename = 'datasets/LIBSVM/cod-rna/cod-rna.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]

    np.random.seed(1)
    ind_train = np.random.choice(len(y), 600, replace=False)
    num, dim = x.shape
    x_train = x[ind_train]
    y_train = y[ind_train]

    print 'Ramp Loss SVM'
    ramp = rampsvm.RampSVM()
    ramp.solve_rampsvm(x_train, y_train)
    ramp.show_result()
    print ramp.weight
