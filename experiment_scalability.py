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

    ## Experimental setup
    sampling_size = np.array([1000, 5000, 10000, 20000, 30000, 40000, 50000, 59535])
    sampling_size = np.array([1000, 5000, 10000, 12000, 15000])
    trial = 1

    ## Arrays for results
    time_ersvm1 = np.zeros([len(sampling_size), trial])
    time_ersvm5 = np.zeros([len(sampling_size), trial])
    time_enusvm1 = np.zeros([len(sampling_size), trial])
    time_enusvm5 = np.zeros([len(sampling_size), trial])
    time_libsvm4 = np.zeros([len(sampling_size), trial])
    time_libsvm0 = np.zeros([len(sampling_size), trial])

    ## Set seed
    np.random.seed(0)

    for i in range(len(sampling_size)):
        for j in range(trial):
            ind_train = np.random.choice(len(y), sampling_size[i], replace=False)
            num, dim = x.shape
            x_train = x[ind_train]
            y_train = y[ind_train]
            initial_weight = np.random.normal(size=dim)

            ## ER-SVM (nu = 0.1)
            print 'ER-SVM'
            ersvm = ersvmdca.LinearPrimalERSVM()
            ersvm.set_nu(0.1)
            ersvm.set_mu(0.05)
            ersvm.set_epsilon(1e-5)
            ersvm.set_cplex_method(1)
            ersvm.set_initial_point(initial_weight, 0)
            ersvm.solve_ersvm(x_train, y_train)
            ersvm.show_result()
            time_ersvm1[i,j] = ersvm.comp_time

            ## ER-SVM (nu = 0.5)
            ersvm.set_nu(0.5)
            ersvm.solve_ersvm(x_train, y_train)
            ersvm.show_result()
            time_ersvm5[i,j] = ersvm.comp_time

            ## Ramp Loss SVM
            ## print 'Ramp Loss SVM'
            ## ramp = rampsvm.RampSVM()
            ## ramp.solve_rampsvm(x_train, y_train)
            ## time_ramp.append(ramp.comp_time)

            ## Enu-SVM (nu = 0.1)
            print 'Enu-SVM'
            enu = enusvm.EnuSVM()
            enu.set_initial_weight(initial_weight)
            enu.set_nu(0.1)
            enu.solve_enusvm(x_train, y_train)
            enu.show_result()
            time_enusvm1[i,j] = enu.comp_time

            ## Enu-SVM (nu = 0.5)
            enu.set_nu(0.5)
            enu.solve_enusvm(x_train, y_train)
            enu.show_result()
            time_enusvm5[i,j] = enu.comp_time

            ## LIBSVM (C = 1e0)
            print 'start libsvm'
            start = time.time()
            clf_libsvm = svm.SVC(C=1e0, kernel='linear')
            clf_libsvm.fit(x_train, y_train)
            end = time.time()
            print 'end libsvm'
            print 'time:', end - start
            time_libsvm0[i,j] = end - start

            ## LIBSVM (C = 1e4)
            print 'start libsvm'
            start = time.time()
            clf_libsvm = svm.SVC(C=1e3, kernel='linear')
            clf_libsvm.fit(x_train, y_train)
            end = time.time()
            print 'end libsvm'
            print 'time:', end - start
            time_libsvm4[i,j] = end - start

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
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['lines.markeredgewidth'] = 0
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    elw = 2
    cs = 5

    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_ersvm1],
                 yerr=[np.std(i) for i in time_ersvm1],
                 label='ER-SVM (nu = 0.1)', elinewidth=elw, capsize=cs, marker='o')
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_ersvm5],
                 yerr=[np.std(i) for i in time_ersvm5],
                 label='ER-SVM (nu = 0.5)', elinewidth=elw, capsize=cs, marker='^')
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_libsvm0],
                 yerr=[np.std(i) for i in time_libsvm0],
                 label='LIBSVM (C = 1e0)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_libsvm4],
                 yerr=[np.std(i) for i in time_libsvm4],
                 label='LIBSVM (C = 1e4)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_enusvm1],
                 yerr=[np.std(i) for i in time_enusvm1],
                 label='Enu-SVM (nu = 0.1)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_enusvm5],
                 yerr=[np.std(i) for i in time_enusvm5],
                 label='Enu-SVM (nu = 0.5)', elinewidth=elw, capsize=cs)
    plt.xlabel('# training samples')
    plt.ylabel('training time (sec)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
