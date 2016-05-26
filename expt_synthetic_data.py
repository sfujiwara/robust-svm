# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn import svm
import time
import logging
import pandas as pd
from mysvm import ersvm, rampsvm, enusvm, ersvmh


# Logging
logging.basicConfig(
    filename="logs/{}.log".format("expt_synthetic_data"),
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("info test")

cov = [[20, 16], [16, 20]]
mu1, mu2 = [3, -3], [-3, 3]


def training_set(r, num_outlier):
    x1 = np.random.multivariate_normal(mu1, cov, 2500)
    x2 = np.random.multivariate_normal(mu2, cov, 2500 - num_outlier)
    y = np.array([1] * 2500 + [-1] * 2500)
    if num_outlier == 0:
        return np.vstack([x1, x2]), y
    outlier = []
    while True:
        tmp = np.random.uniform(-r, r, 2)
        if r < np.linalg.norm(tmp) < r+1 and tmp[0] > tmp[1]:
            outlier.append(tmp)
            if len(outlier) >= num_outlier:
                return np.vstack([x1, x2, np.array(outlier)]), y


def test_set():
    xmat = np.vstack(
        [
            np.random.multivariate_normal(mu1, cov, 2500),
            np.random.multivariate_normal(mu2, cov, 2500)
        ]
    )
    y = np.array([1] * 2500 + [-1] * 2500)
    return xmat, y


np.random.seed(0)
# Fixed hyper-parameters
s = -1
mu = 0.05

dim = 2
# trial = 100
trial = 1000

cost_cand = np.array([5.**i for i in range(4, -5, -1)])
nu_cand = np.linspace(0.9, 0.1, 9)
# num_ol = np.array([0, 1, 2, 3, 4, 5])
# num_ol = np.array([0, 2, 4, 6, 8, 10])
num_ol = np.array([0, 100, 200, 300, 400, 500])

# Initial point generated at random
initial_weight = np.random.normal(size=dim)
initial_weight /= np.linalg.norm(initial_weight)

# DataFrame
df_enu = pd.DataFrame()
df_csvm = pd.DataFrame()
df_var = pd.DataFrame()
df_dca = pd.DataFrame()
df_ramp = pd.DataFrame()
df_ramp_ws = pd.DataFrame()

# Loop for outlier ratio
for i in range(len(num_ol)):

    # Loop for trial
    for j in range(trial):

        print 'Trial:', j
        # Initial point generated at random
        initial_weight = np.random.normal(size=dim)
        initial_weight /= np.linalg.norm(initial_weight)

        # Generate training data and test data
        x_tr, y_tr = training_set(r=75, num_outlier=num_ol[i])
        x_t, y_t = test_set()
        num, dim = x_tr.shape

        # Loop for hyper-parameters
        for k in range(len(nu_cand)):

            # ER-SVM + DCA
            logger.info('Train ER-SVM with DCA (nu, ratio, trial): {}'.format((nu_cand[k], num_ol[i], j)))
            ersvm_clf = ersvm.LinearERSVM(nu=nu_cand[k])
            ersvm_clf.fit(x_tr, y_tr, initial_weight=np.array(initial_weight), initial_bias=0.)
            logger.info('time: {}'.format(ersvm_clf.comp_time))
            row_dca = {
                'outlier_ratio': num_ol[i] / 100.,
                'trial': j,
                'nu': nu_cand[k],
                'test_accuracy': ersvm_clf.score(x_t, y_t),
                'VaR': ersvm_clf.alpha,
                'tr-CVaR': ersvm_clf.obj[-1],
                'comp_time': ersvm_clf.comp_time,
            }
            df_dca = df_dca.append(pd.Series(row_dca, name=pd.datetime.today()))

            # Enu-SVM
            logger.info("Train Enu-SVM (nu, ratio, trial): {}".format((nu_cand[k], num_ol[i], j)))
            clf_enusvm = enusvm.EnuSVM(nu=nu_cand[k])
            clf_enusvm.fit(x_tr, y_tr, np.array(initial_weight))
            logger.info("time: {}".format(clf_enusvm.comp_time))
            row_enu = {
                'outlier_ratio': num_ol[i] / 100.,
                'trial': j,
                'nu': nu_cand[k],
                'test_accuracy': clf_enusvm.score(x_t, y_t),
                'is_convex': clf_enusvm.convexity,
                'comp_time': clf_enusvm.comp_time,
            }
            df_enu = df_enu.append(pd.Series(row_enu, name=pd.datetime.today()))

            # C-SVM
            logger.info("Train C-SVM (C, ratio, trial): {}".format((cost_cand[k], num_ol[i], j)))
            start = time.time()
            clf_csvm = svm.SVC(C=cost_cand[k], kernel="linear", cache_size=2000)
            clf_csvm.fit(x_tr, y_tr)
            end = time.time()
            logger.info("time: {}".format(end-start))
            print 'time:', end - start
            row_csvm = {
                'outlier_ratio': num_ol[i] / 100.,
                'trial': j,
                'C': cost_cand[k],
                'test_accuracy': clf_csvm.score(x_t, y_t),
                'comp_time': end - start,
            }
            df_csvm = df_csvm.append(pd.Series(row_csvm, name=pd.datetime.today()))

            # ER-SVM (Heuristics)
            logger.info("Train ER-SVM with Heuristics (nu, ratio, trial): {}".format((nu_cand[k], num_ol[i], j)))
            clf_var = ersvmh.HeuristicLinearERSVM(nu=nu_cand[k], gamma=0.03/nu_cand[k])
            clf_var.fit(x_tr, y_tr, initial_weight=np.array(initial_weight))
            logger.info("time: {}".format(clf_var.comp_time))
            row_var = {
                'outlier_ratio': num_ol[i] / 100.,
                'trial': j,
                'nu': nu_cand[k],
                'test_accuracy': clf_var.score(x_t, y_t),
                'is_convex': clf_var.is_convex,
                'comp_time': clf_var.comp_time
            }
            df_var = df_var.append(pd.Series(row_var, name=pd.datetime.today()))

            # Ramp Loss SVM
            print 'Start Ramp Loss SVM'
            logger.info('Train ramp loss SVM (C, ratio, trial): {}'.format(cost_cand[k], num_ol[i], j))
            ramp = rampsvm.RampSVM(C=cost_cand[k])
            # ramp.cplex_method = 0  # automatic
            # ramp.set_cost(cost_cand[k])
            ramp.fit(x_tr, y_tr)
            logger.info('time: {}'.format(ramp.comp_time))
            # ramp.show_result()
            row_ramp = {
                'outlier_ratio': num_ol[i] / 100.,
                'trial': j,
                'C': cost_cand[k],
                'test_accuracy': ramp.score(x_t, y_t),
                'comp_time': ramp.comp_time,
                'timeout': ramp.timeout
            }
            df_ramp = df_ramp.append(pd.Series(row_ramp, name=pd.datetime.today()))

logger.info("Training finished")

# Save as csv
# dir_name_result = 'results/synthetic-final/'
dir_name_result = "results/synthetic/num5000_dim2"
if not os.path.isdir(dir_name_result):
    os.makedirs(dir_name_result)
df_dca.to_csv(dir_name_result+'dca.csv', index=False)
df_enu.to_csv(dir_name_result+'enusvm.csv', index=False)
df_var.to_csv(dir_name_result+'var.csv', index=False)
df_ramp.to_csv(dir_name_result+'ramp.csv', index=False)
df_csvm.to_csv(dir_name_result+'csvm.csv', index=False)
df_ramp_ws.to_csv(dir_name_result+'ramp_ws.csv', index=False)
