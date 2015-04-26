import numpy as np
import ersvmdca
import ersvmutil
from sklearn import svm
import matplotlib.pyplot as plt
import time
import ersvm_backup

if __name__ == '__main__':
    ## Read a UCI dataset
    dataset = np.loadtxt('cod-rna.csv', delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    np.random.seed(0)
    times_ersvm = []
    times_libsvm = []
    times_liblinear = []
    for i in range(1, 3):
        ind_train = np.random.choice(len(y), i*1000, replace=False)
        num, dim = x.shape
        ersvm = ersvmdca.LinearPrimalERSVM()
        ersvm.set_nu(0.3)
        ersvm.set_initial_point(2*np.ones(dim)/np.linalg.norm(np.ones(dim)), 0)
        ersvm.solve_ersvm(x[ind_train], y[ind_train])
        ersvm.show_result()
        times_ersvm.append(ersvm.comp_time)
        ## print 'start old version'
        ## start = time.time()
        ## res, eta = ersvm_backup.diff_cvar(x[ind_train], y[ind_train],
        ##                                  2*np.ones(dim)/np.linalg.norm(np.ones(dim)), 0, 0.3, 0.1)
        ## end = time.time()
        ## print 'end old version'
        ## print 'time:', end - start
        ## names_w = ['w%s' % i for i in range(dim)]
        ## print res.solution.get_values(names_w), res.solution.get_values('b')
        ## LIBSVM
        print 'start libsvm'
        start = time.time()
        clf_libsvm = svm.SVC(C=1.0, kernel='linear')
        clf_libsvm.fit(x[ind_train], y[ind_train])
        end = time.time()
        print 'end libsvm'
        print 'time:', end - start
        times_libsvm.append(end - start)
        ## LIBLINEAR
        print 'start liblinear'
        start = time.time()
        clf_libsvm = svm.LinearSVC(C=1.0)
        clf_libsvm.fit(x[ind_train], y[ind_train])
        end = time.time()
        print 'end libsvm'
        print 'time:', end - start
        times_libsvm.append(end - start)

    plt.plot(np.arange(1, 11)*1000, times_ersvm, label='ER-SVM (DCA)')
    plt.plot(np.arange(1, 11)*1000, times_libsvm, label='C-SVC (LIBSVM)')
    plt.plot(np.arange(1, 11)*1000, times_liblinear, label='C-SVC (LIBLINEAR)')
    plt.xlabel('num of training samples')
    plt.ylabel('training time (sec)')
    plt.legend()
    plt.grid()
    plt.show()
