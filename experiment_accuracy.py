import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import time
import pandas as pd

from src import ersvmdca, rampsvm, enusvm, ersvmutil, ersvmh

if __name__ == '__main__':
    filename = 'datasets/LIBSVM/cod-rna/cod-rna.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]

    ## Scaling
    ersvmutil.libsvm_scale(x)

    np.random.seed(0)
    ind_train = np.random.choice(len(y), 1000, replace=False)
    num, dim = x.shape
    x_train = x[ind_train]
    y_train = y[ind_train]

    initial_weight = np.random.normal(size=dim)
    initial_weight = initial_weight / np.linalg.norm(initial_weight)

    for cv in range(1):
        ## Generate synthetic outliers
        
        print 'Start Ramp Loss SVM'
        ramp = rampsvm.RampSVM()
        ramp.solve_rampsvm(x_train, y_train)
        ramp.show_result()

        print 'Start ER-SVM (DCA)'
        ersvm = ersvmdca.LinearPrimalERSVM()
        ersvm.set_initial_point(initial_weight, 0)
        ersvm.set_nu(0.4)
        ersvm.solve_ersvm(x_train, y_train)
        ersvm.show_result()

        print 'Start Enu-SVM'
        enu = enusvm.EnuSVM()
        enu.set_initial_weight(initial_weight)
        enu.solve_enusvm(x_train, y_train)
        enu.show_result()

        print 'Start ER-SVM (Heuristics)'
        her = ersvmh.HeuristicLinearERSVM()
        her.set_initial_weight(initial_weight)
        her.solve_varmin(x_train, y_train)
        her.show_result()

        print 'Start LIBSVM'
        start = time.time()
        libsvm = svm.SVC(C=1.)
        libsvm.fit(x_train, y_train)
        end = time.time()
        print 'End LIBSVM'
        print 'time:', end - start
