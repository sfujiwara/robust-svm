# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
import time
import pandas as pd
from sklearn.metrics import f1_score
from src import ersvmdca, rampsvm, enusvm, ersvmutil, ersvmh
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
    ## Fixed hyper-parameters
    s = -1
    mu = 0.05

    dim = 2
    trial = 100

    
    cost_cand = np.array([5.**i for i in range(4, -5, -1)])
    nu_cand = np.linspace(0.9, 0.1, 9)
    num_ol = np.array([0, 1, 2, 3, 4, 5])
    ## num_ol = np.array([0])

    # Initial point generated at random
    initial_weight = np.random.normal(size=dim)
    initial_weight = initial_weight / np.linalg.norm(initial_weight)

    # Class instances
    ersvm = ersvmdca.LinearPrimalERSVM()
    ersvm.set_initial_point(np.array(initial_weight), 0)
    ersvm.set_mu(0.05)
    ramp = rampsvm.RampSVM()
    ramp.time_limit = 15
    enu = enusvm.EnuSVM()
    var = ersvmh.HeuristicLinearERSVM()
    libsvm = svm.SVC(C=1e0, kernel='linear', max_iter=-1)
    conv_ersvm = ersvmdca.LinearPrimalERSVM()
    conv_ersvm.set_initial_point(np.array(initial_weight), 0)
    conv_ersvm.set_constant_t(0)


    ## DataFrame
    df_enu  = pd.DataFrame()
    df_csvm = pd.DataFrame()
    df_var  = pd.DataFrame()
    df_dca  = pd.DataFrame()
    df_ramp = pd.DataFrame()

    ## Loop for outlier ratio
    for i in range(len(num_ol)):

        ## Loop for trial
        for j in range(trial):

            print 'Trial:', j
            # Initial point generated at random
            initial_weight = np.random.normal(size=dim)
            initial_weight = initial_weight / np.linalg.norm(initial_weight)

            ## Generate training data and test data
            x_tr, y_tr = training_set(r=75, num_outlier=num_ol[i])
            x_t, y_t = test_set()
            num, dim = x_tr.shape

            ## Loop for hyper-parameters
            for k in range(len(nu_cand)):

                ##### ER-SVM + DCA #####
                ersvm.set_nu(nu_cand[k])
                ## ersvm.set_initial_point(np.array(initial_weight_dca), 0)
                ersvm.solve_ersvm(x_tr, y_tr)
                ## ersvm.show_result()
                row_dca = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'nu'           : nu_cand[k],
                    'test_accuracy': ersvm.calc_accuracy(x_t, y_t),
                    'VaR'          : ersvm.alpha,
                    'tr-CVaR'      : ersvm.obj[-1],
                    'comp_time'    : ersvm.comp_time,
                }
                df_dca = df_dca.append(pd.Series(row_dca, name=pd.datetime.today()))
                ersvm.set_initial_point(np.array(ersvm.weight), 0)

                ##### Enu-SVM #####
                enu.set_initial_weight(np.array(initial_weight))
                enu.set_nu(nu_cand[k])
                enu.solve_enusvm(x_tr, y_tr)
                ## enu.show_result()
                row_enu = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'nu'           : nu_cand[k],
                    'test_accuracy': enu.calc_accuracy(x_t, y_t),
                    'is_convex'    : enu.convexity,
                    'comp_time'    : enu.comp_time,
                }
                df_enu = df_enu.append(pd.Series(row_enu, name=pd.datetime.today()))

                ##### C-SVM #####
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

                ##### ER-SVM (Heuristics) #####
                var.set_initial_weight(np.array(initial_weight))
                var.set_nu(nu_cand[k])
                var.set_gamma(0.03/nu_cand[k])
                var.solve_varmin(x_tr, y_tr)
                var.show_result()
                row_var = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'nu'           : nu_cand[k],
                    'test_accuracy': var.calc_accuracy(x_t, y_t),
                    'is_convex'    : var.is_convex,
                    'comp_time'    : var.comp_time
                }
                df_var = df_var.append(pd.Series(row_var, name=pd.datetime.today()))

                ##### Ramp Loss SVM #####
                print 'Start Ramp Loss SVM'
                ramp.cplex_method = 1
                ramp.set_cost(cost_cand[k])
                ramp.solve_rampsvm(x_tr, y_tr)
                ramp.show_result()
                row_ramp = {
                    'outlier_ratio': num_ol[i] / 100.,
                    'trial'        : j,
                    'C'            : cost_cand[k],
                    'test_accuracy': ramp.calc_accuracy_linear(x_t, y_t),
                    'comp_time'    : ramp.comp_time,
                    'timeout'      : ramp.timeout
                }
                df_ramp = df_ramp.append(pd.Series(row_ramp, name=pd.datetime.today()))


    # Save as csv
    dir_name_result = 'results/synthetic/'
    df_dca.to_csv(dir_name_result+'dca.csv', index=False)
    df_enu.to_csv(dir_name_result+'enu.csv', index=False)
    df_var.to_csv(dir_name_result+'var.csv', index=False)
    df_ramp.to_csv(dir_name_result+'ramp.csv', index=False)
    df_csvm.to_csv(dir_name_result+'csvm.csv', index=False)
