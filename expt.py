# -*- coding: utf-8 -*-

# import time
import sys
import os
import logging
import yaml
import argparse
import numpy as np
import pandas as pd
# Import my modules
from mysvm import ersvm, ersvmh, enusvm, rampsvm, svmutil
import data_loader
from sklearn import svm

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conf-file", type=str)
args = parser.parse_args()

# Load Config
config = yaml.load(open(args.conf_file).read())
DATASET_NAME = config["data"]["name"]
OUTLIER_METHOD = config["outlier"]["method"]

# Logging
logging.basicConfig(
    filename="logs/{}.log".format(DATASET_NAME),
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("info test")

# Load internet ad. data (nu_min, nu_max) = (0012, 0.323)
x, y, x_outlier, y_outlier = data_loader.load_data(DATASET_NAME)
num, dim = x.shape

# Set seed
np.random.seed(0)

# Experimental set up
num_tr = int(num * 0.4)   # size of training set
num_val = int(num * 0.3)  # size of validation set
num_t = num - num_tr - num_val    # size of test set
trial = config["num_trials"]

# num_tr = 10

# Candidates of hyper-parameters
nu_list = np.linspace(config["hyper_parameters"]["nu_max"], config["hyper_parameters"]["nu_min"], 9)
c_list = np.array([5. ** i for i in range(4, -5, -1)])
outlier_ratio = np.array([0., 0.03, 0.05, 0.1, 0.15, 0.2])

# sys.exit(0)

# Scaling
# ersvmutil.libsvm_scale(x)
svmutil.standard_scale(x)

# Initial point generated at random
# initial_weight = np.random.normal(size=dim)
# initial_weight /= np.linalg.norm(initial_weight)

# Initial point generated using C-SVM
clf = svm.SVC(kernel="linear")
clf.fit(x, y)
initial_weight = clf.coef_
initial_weight /= np.linalg.norm(initial_weight)

# DataFrame for results
df_ersvm = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'mu', 'val-acc', 'val-f', 'test-acc', 'test-f', 'VaR', 'tr-CVaR', 'comp_time'])
df_var = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'val-acc', 'val-f', 'test-acc', 'test-f', 'is_convex', 'comp_time'])
df_enusvm = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'val-acc', 'val-f', 'test-acc', 'test-f', 'is_convex', 'comp_time'])
df_ramp = pd.DataFrame(columns=['ratio', 'trial', 'C', 's', 'val-acc', 'val-f', 'test-acc', 'test-f', 'comp_time', 'timeout'])
# df_libsvm = pd.DataFrame(columns=['ratio', 'trial', 'C', 'val-acc', 'val-f', 'test-acc', 'test-f', 'comp_time'])

# Loop for outlier ratio
for i in range(len(outlier_ratio)):
    num_ol_tr = int(num_tr * outlier_ratio[i])
    num_ol_val = int(num_val * outlier_ratio[i])
    # Loop for random splitting
    for j in range(trial):
        # Split indices to training, validation, and test set
        ind_rand = np.random.permutation(range(num))
        ind_tr = ind_rand[:num_tr]
        ind_val = ind_rand[num_tr:(num_tr+num_val)]
        ind_t = ind_rand[(num_tr+num_val):]
        # Copy training and validation set since they will be contaminated
        x_tr, y_tr = np.array(x[ind_tr]), np.array(y[ind_tr])      # training set
        x_val, y_val = np.array(x[ind_val]), np.array(y[ind_val])  # validation samples
        # Generate synthetic outliers
        if num_ol_tr > 0:
            # x_tr[:num_ol_tr] = svmutil.runif_sphere(radius=radius, dim=dim, size=num_ol_tr)
            # x_val[:num_ol_val] = svmutil.runif_sphere(radius=radius, dim=dim, size=num_ol_val)
            if OUTLIER_METHOD == "label_flip":
                y_tr[np.random.choice(num_tr, num_ol_tr, replace=False)] *= -1.
                y_val[np.random.choice(num_val, num_ol_val, replace=False)] *= -1.
            elif OUTLIER_METHOD == "another_class":
                ind_ol_tr = np.random.choice(len(x_outlier), num_ol_tr, replace=False)
                ind_ol_val = np.random.choice(len(x_outlier), num_ol_val, replace=False)
                x_tr[:num_ol_tr] = x_outlier[ind_ol_tr]
                x_val[:num_ol_val] = x_outlier[ind_ol_val]
                y_tr[:num_ol_tr], y_val[:num_ol_val] = y_outlier[ind_ol_tr], y_outlier[ind_ol_val]
            else:
                raise ValueError("{} is invalid value as OUTLIER_METHOD".format(OUTLIER_METHOD))
        # Initial point generated at random
        initial_weight = np.random.normal(size=dim)
        initial_weight /= np.linalg.norm(initial_weight)
        # Loop for hyper parameters tuning
        for k in range(len(nu_list)):
            # Ramp Loss SVM
            logger.info('Train ramp loss SVM (C, ratio, trial): {}'.format(c_list[k], outlier_ratio[i], j))
            model_ramp = rampsvm.RampSVM(C=c_list[k])
            model_ramp.fit(x_tr, y_tr)
            logger.info('time: {}'.format(model_ramp.comp_time))
            row_ramp = {
                'ratio': outlier_ratio[i],
                'trial': j,
                'C': c_list[k],
                's': model_ramp.s,
                'val-acc': model_ramp.score(x_val, y_val),
                'val-f': model_ramp.f1_score(x_val, y_val),
                'test-acc': model_ramp.score(x[ind_t], y[ind_t]),
                'test-f': model_ramp.f1_score(x[ind_t], y[ind_t]),
                'comp_time': model_ramp.comp_time,
                'timeout': model_ramp.timeout
            }
            df_ramp = df_ramp.append(pd.Series(row_ramp, name=pd.datetime.today()))
            # ER-SVM using DC Algorithm
            logger.info('Train ER-SVM with DCA (nu, ratio, trial): {}'.format((nu_list[k], outlier_ratio[i], j)))
            model_ersvm = ersvm.LinearERSVM(nu=nu_list[k])
            model_ersvm.fit(x_tr, y_tr, initial_weight)
            logger.info('time: {}'.format(model_ersvm.comp_time))
            row_dca = {
                'ratio': outlier_ratio[i],
                'trial': j,
                'nu': nu_list[k],
                'mu': model_ersvm.mu,
                'val-acc': model_ersvm.score(x_val, y_val),
                'val-f': model_ersvm.f1_score(x_val, y_val),
                'test-acc': model_ersvm.score(x[ind_t], y[ind_t]),
                'test-f': model_ersvm.f1_score(x[ind_t], y[ind_t]),
                'VaR': model_ersvm.alpha,
                'tr-CVaR': model_ersvm.obj[-1],
                'comp_time': model_ersvm.comp_time
            }
            df_ersvm = df_ersvm.append(pd.Series(row_dca, name=pd.datetime.today()))
            # Enu-SVM
            logger.info("Train Enu-SVM (nu, ratio, trial): {}".format((nu_list[k], outlier_ratio[i], j)))
            model_enusvm = enusvm.EnuSVM(nu=nu_list[k])
            model_enusvm.fit(x_tr, y_tr, initial_weight)
            logger.info("time: {}".format(model_enusvm.comp_time))
            row_enusvm = {
                'ratio': outlier_ratio[i],
                'trial': j,
                'nu': nu_list[k],
                'val-acc': model_enusvm.score(x_val, y_val),
                'val-f': model_enusvm.f1_score(x_val, y_val),
                'test-acc': model_enusvm.score(x[ind_t], y[ind_t]),
                'test-f': model_enusvm.f1_score(x[ind_t], y[ind_t]),
                'is_convex': model_enusvm.convexity,
                'comp_time': model_enusvm.comp_time
            }
            df_enusvm = df_enusvm.append(pd.Series(row_enusvm, name=pd.datetime.today()))
            # ER-SVM with heuristic VaR minimization algorithm
            logger.info("Train ER-SVM with Heuristics (nu, ratio, trial): {}".format((nu_list[k], outlier_ratio[i], j)))
            model_var = ersvmh.HeuristicLinearERSVM(nu=nu_list[k], gamma=0.03 / nu_list[k])
            model_var.fit(x_tr, y_tr, initial_weight)
            logger.info("time: {}".format(model_var.comp_time))
            row_var = {
                'ratio': outlier_ratio[i],
                'trial': j,
                'nu': nu_list[k],
                'val-acc': model_var.score(x_val, y_val),
                'val-f': model_var.f1_score(x_val, y_val),
                'test-acc': model_var.score(x[ind_t], y[ind_t]),
                'test-f': model_var.f1_score(x[ind_t], y[ind_t]),
                'is_convex': model_var.is_convex,
                'comp_time': model_var.comp_time
            }
            df_var = df_var.append(pd.Series(row_var, name=pd.datetime.today()))
            # C-SVM (libsvm)
            # print 'Start libsvm'
            # start = time.time()
            # model_libsvm = svm.SVC(C=c_list[k], kernel='linear', max_iter=-1)
            # model_libsvm.fit(x_tr, y_tr)
            # end = time.time()
            # print 'End libsvm'
            # print 'time:', end - start
            # row_libsvm = {
            #     'ratio': outlier_ratio[i],
            #     'trial': j,
            #     'C': c_list[k],
            #     'val-acc': model_libsvm.score(x_val, y_val),
            #     'val-f': f1_score(y_val, model_libsvm.predict(x_val)),
            #     'test-acc': model_libsvm.score(x[ind_t], y[ind_t]),
            #     'test-f': f1_score(y[ind_t], model_libsvm.predict(x[ind_t]))
            # }
            # df_libsvm = df_libsvm.append(pd.Series(row_libsvm, name=pd.datetime.today()))

logger.info("Training finished")
# pd.set_option('line_width', 200)

# Save as csv
if not os.path.isdir("results/{}".format(DATASET_NAME)):
    os.makedirs("results/{}".format(DATASET_NAME))
df_ersvm.to_csv("results/{}/ersvm.csv".format(DATASET_NAME), index=False)
df_enusvm.to_csv("results/{}/enusvm.csv".format(DATASET_NAME), index=False)
df_var.to_csv("results/{}/var.csv".format(DATASET_NAME), index=False)
df_ramp.to_csv("results/{}/ramp.csv".format(DATASET_NAME), index=False)
