# -*- coding: utf-8 -*-

"""
Compare the difference of test accuracy between Extended Robust-SVM and Non-Extended Robust-SVM
using synthetic data
"""

import numpy as np
from sklearn import svm
import time
import pandas as pd
from sklearn.metrics import f1_score
from fsvm import ersvm, rampsvm, enusvm, svmutil, ersvmh
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(0)

    num_train_p = 20
    num_train_n = 480
    num_train_n_ol = 0
    num_test_p = 200
    num_test_n = 4800

    mean_p = np.array([-2, 2])
    mean_n = np.array([0, 0])
    mean_n_ol = np.array([-5, 5])
    cov_p = [[0.5,-0.2], [-0.2,0.5]]
    cov_n = [[1.8, 0], [0, 1.8]]

    nu = 0.051
    mu = 0.01

    model_nonconv = ersvm.LinearERSVM()
    model_nonconv.set_nu(nu)
    model_nonconv.set_mu(mu)
    model_conv = ersvm.LinearERSVM()
    model_conv.set_nu(nu)
    model_conv.set_mu(mu)
    model_conv.set_constant_t(0)

    # Initial point generated at random
    initial_weight = np.random.normal(size=2)
    initial_weight = initial_weight / np.linalg.norm(initial_weight)

    # Set initial point
    model_nonconv.set_initial_point(initial_weight, 0)
    model_conv.set_initial_point(initial_weight, 0)

    res_conv, res_nonconv = [], []

    dist  = np.linspace(4, 2, 5)
    for i in range(len(dist)):
        mean_p = np.array([-1, 1]) * dist[i]
        acc_nonconv = []
        acc_conv = []
        for j in range(10):
            x_train_p = np.random.multivariate_normal(mean=mean_p, cov=cov_p, size=num_train_p)
            x_train_n = np.random.multivariate_normal(mean=mean_n, cov=cov_n, size=num_train_n)
            x_train_n_ol = np.random.multivariate_normal(mean=mean_n_ol, cov=cov_n, size=num_train_n_ol)
            x_train = np.vstack([x_train_p, x_train_n, x_train_n_ol])
            x_test_p = np.random.multivariate_normal(mean=mean_p, cov=cov_p, size=num_test_p)
            x_test_n = np.random.multivariate_normal(mean=mean_n, cov=cov_n, size=num_test_n)
            x_test = np.vstack([x_test_p, x_test_n])
            y_train = np.array([1.]*num_train_p + [-1.]*(num_train_n+num_train_n_ol))
            y_test = np.array([1.]*num_test_p + [-1.]*num_test_n)

            model_nonconv.fit(x_train, y_train)
            model_nonconv.show_result()

            model_conv.fit(x_train, y_train)
            model_conv.show_result()

            acc_nonconv.append(model_nonconv.score(x_test, y_test))
            acc_conv.append(model_conv.score(x_test, y_test))
        res_conv.append(acc_conv)
        res_nonconv.append(acc_nonconv)

    print model_nonconv.score(x_test, y_test)
    print model_conv.score(x_test, y_test)

    ## plt.plot(x_train[:num_train_p, 0], x_train[:num_train_p, 1], 'x')
    ## plt.plot(x_train[num_train_p:, 0], x_train[num_train_p:, 1], '+')
    ## plt.show()

    print acc_nonconv
    print acc_conv

    ## Set parameters
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['lines.linewidth'] = 5
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    x = [np.sqrt(2*i**2) for i in dist]

    plt.plot(x, [np.mean(i) for i in res_nonconv], '-', label='Case N + Case C')
    plt.plot(x, [np.mean(i) for i in res_conv], '--', label='Case C')
    plt.grid()
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Distance')
    plt.show()
