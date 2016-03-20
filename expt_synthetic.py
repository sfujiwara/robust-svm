# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
import time
import pandas as pd
from sklearn.metrics import f1_score
from mysvm import ersvm, rampsvm, enusvm, svmutil, ersvmh
import sys

cov = [[20,16], [16,20]]
mu1, mu2 = [3,-3], [-3,3]


def training_set(r, num_outlier):
    x1 = np.random.multivariate_normal(mu1, cov, 50)
    x2 = np.random.multivariate_normal(mu2, cov, 50 - num_outlier)
    y = np.array([1] * 50 + [-1] * 50)
    if num_outlier == 0: return np.vstack([x1, x2]), y
    outlier = []
    while True:
        tmp = np.random.uniform(-r,r,2)
        if r < np.linalg.norm(tmp) < r+1 and tmp[0] > tmp[1]:
            outlier.append(tmp)
            if len(outlier) >= num_outlier:
                return np.vstack([x1, x2, np.array(outlier)]), y


def test_set():
    xmat = np.vstack([np.random.multivariate_normal(mu1, cov, 1000),
                           np.random.multivariate_normal(mu2, cov, 1000)])
    y = np.array([1] * 1000 + [-1] * 1000)
    return xmat, y


if __name__ == '__main__':
    np.random.seed(0)
    # Fixed hyper-parameters
    s = -1
    mu = 0.05

    dim = 2
    trial = 100

    cost_cand = np.array([5.**i for i in range(4, -5, -1)])
    nu_cand = np.linspace(0.9, 0.1, 9)
    # num_ol = np.array([0, 1, 2, 3, 4, 5])
    num_ol = np.array([0, 2, 4, 6, 8, 10])

    # Initial point generated at random
    initial_weight = np.random.normal(size=dim)
    initial_weight = initial_weight / np.linalg.norm(initial_weight)

    # Class instances
    ersvm = ersvm.LinearERSVM()
    ersvm.set_initial_point(np.array(initial_weight), 0)
    ersvm.set_mu(0.05)
    ramp = rampsvm.RampSVM()
    ramp.time_limit = 15
    enu = enusvm.EnuSVM()
    var = ersvmh.HeuristicLinearERSVM()
    libsvm = svm.SVC(C=1e0, kernel='linear', max_iter=-1)
    conv_ersvm = ersvm.LinearERSVM()
    conv_ersvm.set_initial_point(np.array(initial_weight), 0)
    conv_ersvm.set_constant_t(0)
    ramp_ws = rampsvm.RampSVM()
    ramp_ws.time_limit = 15

    # DataFrame
    df_enu  = pd.DataFrame()
    df_csvm = pd.DataFrame()
    df_var  = pd.DataFrame()
    df_dca  = pd.DataFrame()
    df_ramp = pd.DataFrame()
    df_ramp_ws = pd.DataFrame()

    # Loop for outlier ratio
    for i in range(len(num_ol)):

        # Loop for trial
        for j in range(trial):

            print 'Trial:', j
            # Initial point generated at random
            initial_weight = np.random.normal(size=dim)
            initial_weight = initial_weight / np.linalg.norm(initial_weight)

            # Generate training data and test data
            x_tr, y_tr = training_set(r=75, num_outlier=num_ol[i])
            x_t, y_t = test_set()
            num, dim = x_tr.shape

            # Loop for hyper-parameters
            for k in range(len(nu_cand)):

                # ER-SVM + DCA
                ersvm.set_nu(nu_cand[k])
                ersvm.set_cplex_method(0)  # automatic
                # ersvm.set_initial_point(np.array(initial_weight_dca), 0)
                ersvm.fit(x_tr, y_tr)
                # ersvm.show_result()
                row_dca = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'nu'           : nu_cand[k],
                    'test_accuracy': ersvm.score(x_t, y_t),
                    'VaR'          : ersvm.alpha,
                    'tr-CVaR'      : ersvm.obj[-1],
                    'comp_time'    : ersvm.comp_time,
                }
                df_dca = df_dca.append(pd.Series(row_dca, name=pd.datetime.today()))
                ersvm.set_initial_point(np.array(ersvm.weight), 0)

                # Enu-SVM
                enu.set_initial_weight(np.array(initial_weight))
                enu.set_nu(nu_cand[k])
                enu.fit(x_tr, y_tr)
                # enu.show_result()
                row_enu = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'nu'           : nu_cand[k],
                    'test_accuracy': enu.score(x_t, y_t),
                    'is_convex'    : enu.convexity,
                    'comp_time'    : enu.comp_time,
                }
                df_enu = df_enu.append(pd.Series(row_enu, name=pd.datetime.today()))

                # C-SVM
                start = time.time()
                libsvm.set_params(**{'C': cost_cand[k]})
                libsvm.fit(x_tr, y_tr)
                end = time.time()
                print 'time:', end - start
                row_csvm = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'C'            : cost_cand[k],
                    'test_accuracy': libsvm.score(x_t, y_t),
                    'comp_time'    : end - start,
                }
                df_csvm = df_csvm.append(pd.Series(row_csvm, name=pd.datetime.today()))

                # ER-SVM (Heuristics)
                var.set_initial_weight(np.array(initial_weight))
                var.set_nu(nu_cand[k])
                var.set_gamma(0.03/nu_cand[k])
                var.fit(x_tr, y_tr)
                var.show_result()
                row_var = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'nu'           : nu_cand[k],
                    'test_accuracy': var.score(x_t, y_t),
                    'is_convex'    : var.is_convex,
                    'comp_time'    : var.comp_time
                }
                df_var = df_var.append(pd.Series(row_var, name=pd.datetime.today()))

                # Ramp Loss SVM
                print 'Start Ramp Loss SVM'
                ramp.cplex_method = 0  # automatic
                ramp.set_cost(cost_cand[k])
                ramp.fit(x_tr, y_tr)
                ramp.show_result()
                row_ramp = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'C'            : cost_cand[k],
                    'test_accuracy': ramp.score(x_t, y_t),
                    'comp_time'    : ramp.comp_time,
                    'timeout'      : ramp.timeout
                }
                df_ramp = df_ramp.append(pd.Series(row_ramp, name=pd.datetime.today()))

                # Ramp Loss SVM
                print 'Start Ramp Loss SVM'
                ramp_ws.cplex_method = 1  # primal simplex (with warm start)
                ramp_ws.set_cost(cost_cand[k])
                ramp_ws.fit(x_tr, y_tr)
                ramp_ws.show_result()
                row_ramp_ws = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'C'            : cost_cand[k],
                    'test_accuracy': ramp_ws.score(x_t, y_t),
                    'comp_time'    : ramp_ws.comp_time,
                    'timeout'      : ramp_ws.timeout
                }
                df_ramp_ws = df_ramp_ws.append(pd.Series(row_ramp_ws, name=pd.datetime.today()))

    # Save as csv
    dir_name_result = 'results/synthetic-final/'
    df_dca.to_csv(dir_name_result+'dca.csv', index=False)
    df_enu.to_csv(dir_name_result+'enusvm.csv', index=False)
    df_var.to_csv(dir_name_result+'var.csv', index=False)
    df_ramp.to_csv(dir_name_result+'ramp.csv', index=False)
    df_csvm.to_csv(dir_name_result+'csvm.csv', index=False)
    df_ramp_ws.to_csv(dir_name_result+'ramp_ws.csv', index=False)
