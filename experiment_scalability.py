import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import time
import pandas as pd

from src import ersvmdca, rampsvm, enusvm, ersvmutil

if __name__ == '__main__':
    ## Read a UCI dataset
    filename = 'datasets/LIBSVM/cod-rna/cod-rna.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    ind_neg = np.where(y==2)[0]
    y[ind_neg] = -1

    ## Scaling
    ## ersvmutil.libsvm_scale(x)

    sampling_size = np.array([1000, 5000, 10000, 20000, 30000, 40000, 50000, 59535])
    sampling_size = np.array([1000, 5000, 10000, 12000, 15000])

    ## Set seed
    np.random.seed(0)

    time_ersvm1 = []
    time_ersvm5 = []
    time_ramp = []
    time_enusvm1 = []
    time_enusvm5 = []
    time_libsvm0 = []
    time_libsvm4 = []
    time_liblinear = []

    for i in sampling_size:
        ind_train = np.random.choice(len(y), i, replace=False)
        num, dim = x.shape
        x_train = x[ind_train]
        y_train = y[ind_train]
        initial_weight = np.random.normal(size=dim)

        ## ER-SVM
        print 'ER-SVM'
        ersvm = ersvmdca.LinearPrimalERSVM()
        ersvm.set_nu(0.1)
        ersvm.set_mu(0.05)
        ersvm.set_epsilon(1e-5)
        ersvm.set_cplex_method(1)
        ersvm.set_initial_point(initial_weight, 0)
        ersvm.solve_ersvm(x_train, y_train)
        ersvm.show_result()
        time_ersvm1.append(ersvm.comp_time)
        ersvm.set_nu(0.5)
        ersvm.solve_ersvm(x_train, y_train)
        ersvm.show_result()
        time_ersvm5.append(ersvm.comp_time)

        ## Ramp Loss SVM
        ## print 'Ramp Loss SVM'
        ## ramp = rampsvm.RampSVM()
        ## ramp.solve_rampsvm(x_train, y_train)
        ## time_ramp.append(ramp.comp_time)

        ## Enu-SVM
        print 'Enu-SVM'
        enu = enusvm.EnuSVM()
        enu.set_initial_weight(initial_weight)
        enu.set_nu(0.1)
        enu.solve_enusvm(x_train, y_train)
        time_enusvm1.append(enu.comp_time)
        enu.show_result()
        enu.set_nu(0.5)
        enu.solve_enusvm(x_train, y_train)
        time_enusvm5.append(enu.comp_time)
        enu.show_result()

        ## LIBSVM
        print 'start libsvm'
        start = time.time()
        clf_libsvm = svm.SVC(C=1e0, kernel='linear')
        clf_libsvm.fit(x_train, y_train)
        end = time.time()
        print 'end libsvm'
        print 'time:', end - start
        time_libsvm0.append(end - start)
        print 'start libsvm'
        start = time.time()
        clf_libsvm = svm.SVC(C=1e3, kernel='linear')
        clf_libsvm.fit(x_train, y_train)
        end = time.time()
        print 'end libsvm'
        print 'time:', end - start
        time_libsvm4.append(end - start)

        ## LIBLINEAR
        ## print 'start liblinear'
        ## start = time.time()
        ## clf_liblinear = svm.LinearSVC(C=1.0)
        ## clf_liblinear.fit(x_train, y_train)
        ## end = time.time()
        ## print 'end liblinear'
        ## print 'time:', end - start
        ## time_liblinear.append(end - start)

    ## Set parameters for plot
    params = {'axes.labelsize': 24,
              'lines.linewidth' : 5,
              ## 'text.fontsize': 18,
              'legend.fontsize': 20,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              }
    plt.rcParams.update(params)

    plt.plot(sampling_size, time_ersvm1, label='ER-SVM (nu = 0.1)')
    plt.plot(sampling_size, time_ersvm5, label='ER-SVM (nu = 0.5)')
    plt.plot(sampling_size, time_libsvm0, label='LIBSVM (C = 1e0)')
    plt.plot(sampling_size, time_libsvm4, label='LIBSVM (C = 1e4)')
    ## plt.plot(sampling_size, time_liblinear, label='C-SVC (LIBLINEAR)')
    plt.plot(sampling_size, time_enusvm1, label='Enu-SVM (nu = 0.1)')
    plt.plot(sampling_size, time_enusvm5, label='Enu-SVM (nu = 0.5)')
    ## plt.plot(sampling_size, time_ramp, label='Ramp')
    plt.xlabel('num of training samples')
    plt.ylabel('training time (sec)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
