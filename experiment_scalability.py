## -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import time
import pandas as pd

from src import ersvmdca, rampsvm, enusvm, ersvmh, ersvmutil

if __name__ == '__main__':
    ## Read a UCI dataset
    filename = 'datasets/LIBSVM/cod-rna/cod-rna.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    ind_neg = np.where(y==2)[0]
    y[ind_neg] = -1

    ## Scaling
    #ersvmutil.libsvm_scale(x)
    ersvmutil.standard_scale(x)

    ## Experimental setup
    sampling_size = np.array([1000, 5000, 10000, 20000, 30000, 40000, 50000, 59535])
    ## sampling_size = np.array([500, 1000, 2000, 3000, 5000, 10000])
    trial = 10
    ## trial = 1

    time_limit = 2000
    is_on_time_ersvm1 = True
    is_on_time_ersvm5 = True
    is_on_time_enusvm1 = True
    is_on_time_enusvm5 = True
    is_on_time_libsvm1e0 = True
    is_on_time_libsvm1e3 = True
    is_on_time_var1 = True
    is_on_time_var5 = True

    ## Arrays for results
    time_ersvm1 = np.zeros([len(sampling_size), trial])
    time_ersvm5 = np.zeros([len(sampling_size), trial])
    time_enusvm1 = np.zeros([len(sampling_size), trial])
    time_enusvm5 = np.zeros([len(sampling_size), trial])
    time_libsvm4 = np.zeros([len(sampling_size), trial])
    time_libsvm0 = np.zeros([len(sampling_size), trial])
    time_var1 = np.zeros([len(sampling_size), trial])
    time_var5 = np.zeros([len(sampling_size), trial])
    convexity_enu1 = np.zeros([len(sampling_size), trial])
    convexity_enu5 = np.zeros([len(sampling_size), trial])
    
    ## Set seed
    np.random.seed(0)

    for i in range(len(sampling_size)):
        for j in range(trial):
            print 'sample size', sampling_size[i]
            print 'trial', j
            
            ind_train = np.random.choice(len(y), sampling_size[i], replace=False)
            num, dim = x.shape
            x_train = x[ind_train]
            y_train = y[ind_train]
            initial_weight = np.random.normal(size=dim)

            ## ER-SVM (nu = 0.1)
            print '----- ER-SVM -----'
            print 'nu: 0.1', '#sample:', sampling_size[i], 'trial:', j
            if is_on_time_ersvm1:
                ersvm = ersvmdca.LinearPrimalERSVM()
                ersvm.set_nu(0.1)
                ersvm.set_mu(0.05)
                ersvm.set_epsilon(1e-5)
                ersvm.set_cplex_method(1)
                ersvm.set_initial_point(initial_weight, 0)
                ersvm.solve_ersvm(x_train, y_train)
                ersvm.show_result()
                time_ersvm1[i,j] = ersvm.comp_time
                ## if ersvm.comp_time > time_limit:
                ##     is_on_time_ersvm1 = False

            ## ER-SVM (nu = 0.5)
            print '----- ER-SVM -----'
            print 'nu: 0.5', '#sample:', sampling_size[i], 'trial:', j
            ersvm.set_nu(0.5)
            ersvm.solve_ersvm(x_train, y_train)
            ersvm.show_result()
            time_ersvm5[i,j] = ersvm.comp_time

            ## Heuristic VaR minimization (nu = 0.1)
            print '----- Heuristics -----'
            print 'nu: 0.1', '#sample:', sampling_size[i], 'trial:', j
            if is_on_time_var1:
                var = ersvmh.HeuristicLinearERSVM()
                var.set_nu(0.1)
                var.set_gamma(0.03/0.1)
                var.set_initial_weight(initial_weight)
                var.solve_varmin(x_train, y_train)
                var.show_result()
                time_var1[i,j] = var.comp_time
                if time_var1[i,j] > time_limit:
                    is_on_time_var1 = False

            ## Heuristic VaR minimization (nu = 0.5)
            ## var = ersvmh.HeuristicLinearERSVM()
            print '----- Heuristics -----'
            print 'nu: 0.5', '#sample:', sampling_size[i], 'trial:', j
            if is_on_time_var5:
                var.set_nu(0.5)
                var.set_gamma(0.03/0.5)
                var.set_initial_weight(initial_weight)
                var.solve_varmin(x_train, y_train)
                var.show_result()
                time_var5[i,j] = var.comp_time
                if time_var5[i,j] > time_limit:
                    is_on_time_var5 = False

            ## Ramp Loss SVM
            ## print 'Ramp Loss SVM'
            ## ramp = rampsvm.RampSVM()
            ## ramp.solve_rampsvm(x_train, y_train)
            ## time_ramp.append(ramp.comp_time)

            ## Enu-SVM (nu = 0.1)
            print '----- Enu-SVM -----'
            print 'nu: 0.1', '#sample:', sampling_size[i], 'trial:', j
            enu = enusvm.EnuSVM()
            enu.set_initial_weight(initial_weight)
            enu.set_nu(0.1)
            enu.solve_enusvm(x_train, y_train)
            enu.show_result()
            time_enusvm1[i,j] = enu.comp_time
            convexity_enu1[i,j] = enu.convexity

            ## Enu-SVM (nu = 0.5)
            print '----- Enu-SVM -----'
            print 'nu: 0.5', '#sample:', sampling_size[i], 'trial:', j
            enu.set_nu(0.5)
            enu.solve_enusvm(x_train, y_train)
            enu.show_result()
            time_enusvm5[i,j] = enu.comp_time
            convexity_enu5[i,j] = enu.convexity

            ## LIBSVM (C = 1e0)
            print '----- LIBSVM -----'
            print 'C: 1e0', '#sample:', sampling_size[i], 'trial:', j
            start = time.time()
            clf_libsvm = svm.SVC(C=1e0, kernel='linear')
            clf_libsvm.fit(x_train, y_train)
            end = time.time()
            print 'end libsvm'
            print 'time:', end - start
            time_libsvm0[i,j] = end - start

            ## LIBSVM (C = 1e3)
            if is_on_time_libsvm1e3:
                print '----- LIBSVM -----'
                print 'C: 1e0', '#sample:', sampling_size[i], 'trial:', j
                start = time.time()
                clf_libsvm = svm.SVC(C=1e3, kernel='linear')
                clf_libsvm.fit(x_train, y_train)
                end = time.time()
                print 'end libsvm'
                print 'time:', end - start
                time_libsvm4[i,j] = end - start
                if time_libsvm4[i,j] > time_limit:
                    is_on_time_libsvm1e3 = False

            ## LIBLINEAR
            ## print 'start liblinear'
            ## start = time.time()
            ## clf_liblinear = svm.LinearSVC(C=1.0)
            ## clf_liblinear.fit(x_train, y_train)
            ## end = time.time()
            ## print 'end liblinear'
            ## print 'time:', end - start
            ## time_liblinear.append(end - start)


    ## Save results
    np.savetxt('results/scalability/sampling_size.csv', sampling_size, fmt='%.0f', delimiter=',')
    np.savetxt('results/scalability/ersvm_nu01.csv', time_ersvm1, fmt='%0.9f', delimiter=',')
    np.savetxt('results/scalability/ersvm_nu05.csv', time_ersvm5, fmt='%0.9f', delimiter=',')
    np.savetxt('results/scalability/libsvm_c1e0.csv', time_libsvm0, fmt='%0.9f', delimiter=',')
    np.savetxt('results/scalability/libsvm_c1e3.csv', time_libsvm4, fmt='%0.9f', delimiter=',')
    np.savetxt('results/scalability/enusvm_nu01.csv', time_enusvm1, fmt='%0.9f', delimiter=',')
    np.savetxt('results/scalability/enusvm_nu05.csv', time_enusvm5, fmt='%0.9f', delimiter=',')
    np.savetxt('results/scalability/var_nu01.csv', time_var1, fmt='%0.9f', delimiter=',')
    np.savetxt('results/scalability/var_nu05.csv', time_var5, fmt='%0.9f', delimiter=',')
    np.savetxt('results/scalability/convexity_enusvm_nu01.csv', convexity_enu1, fmt='%0.1f', delimiter=',')
    np.savetxt('results/scalability/convexity_enusvm_nu05.csv', convexity_enu5, fmt='%0.1f', delimiter=',')

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
                 label='LIBSVM (C = 1e3)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_enusvm1],
                 yerr=[np.std(i) for i in time_enusvm1],
                 label='Enu-SVM (nu = 0.1)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_enusvm5],
                 yerr=[np.std(i) for i in time_enusvm5],
                 label='Enu-SVM (nu = 0.5)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_var1],
                 yerr=[np.std(i) for i in time_var1],
                 label='Heuristics (nu = 0.1)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_var5],
                 yerr=[np.std(i) for i in time_var5],
                 label='Heuristics (nu = 0.5)', elinewidth=elw, capsize=cs)
    plt.xlabel('# training samples')
    plt.ylabel('training time (sec)')
    plt.legend(loc='upper left')
    ## plt.ylim([-100, 1500])
    plt.grid()
    plt.show()
