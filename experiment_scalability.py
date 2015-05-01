import numpy as np
import ersvmdca
import ersvmutil
from sklearn import svm
import matplotlib.pyplot as plt
import time
import ersvm_backup
import pandas as pd

if __name__ == '__main__':
    ## Read a UCI dataset
    dataset = np.loadtxt('cod-rna.csv', delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    ind_neg = np.where(y==2)[0]
    y[ind_neg] = -1

    ## Scaling
    for i in xrange(len(x[0])):
        x[:,i] -= np.mean(x[:,i])
        x[:,i] /= np.std(x[:,i])

    ## Set seed
    np.random.seed(0)

    times_ersvm = []
    times_libsvm = []
    times_liblinear = []

    for i in range(1, 11):
        ind_train = np.random.choice(len(y), i*100, replace=False)
        num, dim = x.shape
        ## dim = 2
        x_train = x[ind_train, 0:dim]
        y_train = y[ind_train]
        ## initial_weight = 2*np.ones(dim)/np.linalg.norm(np.ones(dim))
        initial_weight = np.random.normal(size=dim)

        ersvm = ersvmdca.LinearPrimalERSVM()
        ersvm.set_nu(0.5)
        ersvm.set_epsilon(1e-5)
        ersvm.set_cplex_method(1)
        ersvm.set_initial_point(initial_weight, 0)
        ersvm.solve_ersvm(x_train, y_train)
        ersvm.show_result()
        times_ersvm.append(ersvm.comp_time)

        ## print 'start old version'
        ## start = time.time()
        ## res, eta = ersvm_backup.diff_cvar(x_train, y_train,
        ##                                   initial_weight, 0, 0.5, 0.1)
        ## end = time.time()
        ## print 'end old version'
        ## print 'time:', end - start
        ## names_w = ['w%s' % i for i in range(dim)]
        ## print res.solution.get_values(names_w), res.solution.get_values('b')

        ## LIBSVM
        print 'start libsvm'
        start = time.time()
        clf_libsvm = svm.SVC(C=1.0, kernel='linear')
        clf_libsvm.fit(x_train, y_train)
        end = time.time()
        print 'end libsvm'
        print 'time:', end - start
        times_libsvm.append(end - start)

        ## LIBLINEAR
        print 'start liblinear'
        start = time.time()
        clf_liblinear = svm.LinearSVC(C=1.0)
        clf_liblinear.fit(x_train, y_train)
        end = time.time()
        print 'end liblinear'
        print 'time:', end - start
        times_liblinear.append(end - start)

    plt.plot(np.arange(1, 11)*1000, times_ersvm, label='ER-SVM (DCA)')
    plt.plot(np.arange(1, 11)*1000, times_libsvm, label='C-SVC (LIBSVM)')
    plt.plot(np.arange(1, 11)*1000, times_liblinear, label='C-SVC (LIBLINEAR)')
    plt.xlabel('num of training samples')
    plt.ylabel('training time (sec)')
    plt.legend()
    plt.grid()
    plt.show()
