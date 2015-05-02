import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import time
import pandas as pd

from src import ersvmdca, rampsvm, enusvm

if __name__ == '__main__':
    ## Read a UCI dataset
    filename = 'datasets/LIBSVM/cod-rna/cod-rna.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    ind_neg = np.where(y==2)[0]
    y[ind_neg] = -1

    ## Scaling
    ## for i in xrange(len(x[0])):
    ##     x[:,i] -= np.mean(x[:,i])
    ##     x[:,i] /= np.std(x[:,i])

    max_size = 10
    sampling_size = 1000
    ## Set seed
    np.random.seed(0)

    time_ersvm = []
    time_ramp = []
    time_enusvm = []
    time_libsvm = []
    time_liblinear = []
    df_time = pd.DataFrame({'#sample':[], 'ER-SVM':[], 'Ramp-SVM':[],
                            'Enu-SVM':[], 'LIBSVM':[], 'LIBLINEAR':[]})
    s = pd.Series([1,3,5,np.nan,6,8])
    df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'data2' : range(3)})

    for i in range(1, max_size):
        ind_train = np.random.choice(len(y), i*sampling_size, replace=False)
        num, dim = x.shape
        ## dim = 2
        x_train = x[ind_train, 0:dim]
        y_train = y[ind_train]
        initial_weight = np.random.normal(size=dim)

        ## ER-SVM
        print 'ER-SVM'
        ersvm = ersvmdca.LinearPrimalERSVM()
        ersvm.set_nu(0.5)
        ersvm.set_epsilon(1e-5)
        ersvm.set_cplex_method(1)
        ersvm.set_initial_point(initial_weight, 0)
        ersvm.solve_ersvm(x_train, y_train)
        ersvm.show_result()
        time_ersvm.append(ersvm.comp_time)

        ## Ramp Loss SVM
        print 'Ramp Loss SVM'
        ramp = rampsvm.RampSVM()
        ramp.solve_rampsvm(x_train, y_train)
        time_ramp.append(ramp.comp_time)

        ## Enu-SVM
        print 'Enu-SVM'
        enu = enusvm.EnuSVM()
        enu.set_initial_weight(initial_weight)
        enu.solve_enusvm(x_train, y_train)
        time_enusvm.append(enu.comp_time)

        ## LIBSVM
        ## print 'start libsvm'
        ## start = time.time()
        ## clf_libsvm = svm.SVC(C=1.0, kernel='linear')
        ## clf_libsvm.fit(x_train, y_train)
        ## end = time.time()
        ## print 'end libsvm'
        ## print 'time:', end - start
        ## time_libsvm.append(end - start)

        ## LIBLINEAR
        print 'start liblinear'
        start = time.time()
        clf_liblinear = svm.LinearSVC(C=1.0)
        clf_liblinear.fit(x_train, y_train)
        end = time.time()
        print 'end liblinear'
        print 'time:', end - start
        time_liblinear.append(end - start)

    plt.plot(np.arange(1, max_size)*sampling_size, time_ersvm, label='ER-SVM (DCA)')
    ## plt.plot(np.arange(1, max_size)*sampling_size, time_libsvm, label='C-SVC (LIBSVM)')
    plt.plot(np.arange(1, max_size)*sampling_size, time_liblinear, label='C-SVC (LIBLINEAR)')
    plt.plot(np.arange(1, max_size)*sampling_size, time_enusvm, label='Enu-SVM')
    plt.plot(np.arange(1, max_size)*sampling_size, time_ramp, label='Ramp')
    plt.xlabel('num of training samples')
    plt.ylabel('training time (sec)')
    plt.legend()
    plt.grid()
    plt.show()
