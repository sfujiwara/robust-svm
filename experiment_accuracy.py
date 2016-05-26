import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import time
import pandas as pd

from mysvm import ersvm, rampsvm, enusvm, svmutil, ersvmh
#from src_old import ersvm

if __name__ == '__main__':
    ## Set seed
    np.random.seed(0)

    ## Read data set
    ## filename = 'datasets/libsvm/cod-rna/cod-rna.csv'
    ## filename = 'datasets/libsvm/heart/heart_scale.csv'
    filename = 'data/libsvm/liver-disorders/liver-disorders_scale.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    ## ind = np.random.choice(59535, 1000, replace=False)
    ## y = dataset[ind, 0]
    ## x = dataset[ind, 1:]
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape

    ## Scaling
    ## ersvmutil.libsvm_scale(x)
    svmutil.standard_scale(x)

    ## Initial point generated at random
    initial_weight = np.random.normal(size=dim)
    initial_weight = initial_weight / np.linalg.norm(initial_weight)

    ## The number of outliers
    num_outliers = 0

    ## Candidates of hyper-parameters
    ## nu_cand = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    nu_cand = np.linspace(0.75, 0.1, 9)
    cost_cand = np.array([5.**i for i in range(4, -5, -1)])
    outlier_ratio = np.array([0, 0.01, 0.05, 0.1, 0.2])
    ## mu_cand = np.array([0.01, 0.05, 0.1, 0.2])
    ## s_cand = np.array([-1, 0., 0.5])

    ## Class instances
    ersvm = ersvm.LinearERSVM()
    ersvm.set_initial_point(initial_weight, 0)
    ramp = rampsvm.RampSVM()
    enu = enusvm.EnuSVM()
    var = ersvmh.HeuristicLinearERSVM()
    libsvm = svm.SVC(C=1e0, kernel='linear')

    ## For cross validation
    cross = 10
    block_size = int(num/cross)
    ind_cv = np.arange(num)
    np.random.shuffle(ind_cv)

    ## Lists for accuracy
    acc_dca = np.zeros([len(nu_cand), cross])
    acc_var = np.zeros([len(nu_cand), cross])
    acc_enu = np.zeros([len(nu_cand), cross])
    acc_libsvm = np.zeros([len(nu_cand), cross])
    acc_ramp = np.zeros([len(nu_cand), cross])

    for i in range(len(nu_cand)):
        for cv in range(10):
            ## ind_train = np.random.choice(len(y), 500, replace=False)
            ind_test = ind_cv[(block_size * cv):(block_size * (cv+1))]
            ind_train = np.array([k for k in ind_cv if k not in ind_test])
            num, dim = x.shape
            x_train = x[ind_train]
            y_train = y[ind_train]
            x_test = x[ind_test]
            y_test = y[ind_test]

            ## Generate synthetic outliers
            if num_outliers > 0:
                outliers = svmutil.runif_sphere(radius=20, dim=dim, size=num_outliers)
                x_train = np.vstack([x_train, outliers])
                y_train = np.hstack([y_train, np.ones(num_outliers)])

            print 'Start Ramp Loss SVM'
            ramp.set_cost(cost_cand[i])
            ramp.fit(x_train, y_train)
            ramp.show_result()
            acc_ramp[i, cv] = sum((np.dot(x_test, ramp.weight) + ramp.bias) * y_test > 0) / float(len(y_test))

            print 'Start ER-SVM (DCA)'
            ersvm.set_nu(nu_cand[i])
            ersvm.set_mu(0.05)
            ersvm.fit(x_train, y_train)
            ersvm.show_result()
            acc_dca[i, cv] = ersvm.score(x_test, y_test)
            ersvm.set_initial_point(ersvm.weight, 0)

            print 'Start Enu-SVM'
            enu.set_initial_weight(initial_weight)
            enu.set_nu(nu_cand[i])
            enu.fit(x_train, y_train)
            enu.show_result()
            acc_enu[i, cv] = enu.score(x_test, y_test)

            print 'Start ER-SVM (Heuristics)'
            var.set_initial_weight(initial_weight)
            var.set_nu(nu_cand[i])
            var.fit(x_train, y_train)
            var.show_result()
            acc_var[i, cv] = var.score(x_test, y_test)

            print 'Start libsvm'
            start = time.time()
            libsvm.set_params(**{'C':cost_cand[i]})
            libsvm.fit(x_train, y_train)
            acc_libsvm[i, cv] = libsvm.score(x_test, y_test)
            end = time.time()
            print 'End libsvm'
            print 'time:', end - start
