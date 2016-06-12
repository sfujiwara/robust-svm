# -*- coding: utf-8 -*-

# import time
import sys
import os
import time
import logging
import yaml
import argparse
import numpy as np
import pandas as pd
# Import my modules
from mysvm import ersvm, ersvmh, enusvm, rampsvm, svmutil
import data_loader
from sklearn import svm
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conf-file", type=str)
args = parser.parse_args()

# Load Config
config = yaml.load(open(args.conf_file).read())
DATASET_NAME = config["data"]["name"]
OUTLIER_METHOD = config["outlier"]["method"]
radius = config["outlier"]["radius"]

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

# Candidates of hyper-parameters
nu_list = np.linspace(config["hyper_parameters"]["nu_max"], config["hyper_parameters"]["nu_min"], 9)
c_list = np.array([5. ** i for i in range(4, -5, -1)])
outlier_ratio = np.array([0., 0.03, 0.05, 0.1, 0.15, 0.2])

# sys.exit(0)

# Scaling
# ersvmutil.libsvm_scale(x)
svmutil.standard_scale(x)

# Initial point generated at random
initial_weight = np.random.normal(size=dim)
initial_weight /= np.linalg.norm(initial_weight)

# DataFrame for results
df_libsvm = pd.DataFrame(columns=['ratio', 'trial', 'C', 'val-acc', 'val-f', 'test-acc', 'test-f', 'comp_time'])

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
            elif OUTLIER_METHOD == "hyper_sphere":
                outliers = svmutil.runif_sphere(radius=radius, dim=dim, size=num_ol_tr)
                x_tr[:num_ol_tr] = outliers
                outliers = svmutil.runif_sphere(radius=radius, dim=dim, size=num_ol_val)
                x_val[:num_ol_val] = outliers
            else:
                raise ValueError("{} is invalid value as OUTLIER_METHOD".format(OUTLIER_METHOD))
        # Loop for hyper parameters tuning
        for k in range(len(nu_list)):
            # Ramp Loss SVM
            logger.info('Train C-SVM (C, ratio, trial): ({0}, {1}, {2})'.format(c_list[k], outlier_ratio[i], j))
            # C-SVM (libsvm)
            print 'Start libsvm'
            start = time.time()
            clf_csvm = svm.SVC(C=c_list[k], kernel='linear', max_iter=1000000)
            clf_csvm.fit(x_tr, y_tr)
            end = time.time()
            print 'End libsvm'
            print 'time:', end - start
            logger.info('time: {0}, status: {1}'.format(end - start, clf_csvm.fit_status_))
            row_libsvm = {
                'ratio': outlier_ratio[i],
                'trial': j,
                'C': c_list[k],
                'val-acc': clf_csvm.score(x_val, y_val),
                'val-f': f1_score(y_val, clf_csvm.predict(x_val)),
                'test-acc': clf_csvm.score(x[ind_t], y[ind_t]),
                'test-f': f1_score(y[ind_t], clf_csvm.predict(x[ind_t]))
            }
            df_libsvm = df_libsvm.append(pd.Series(row_libsvm, name=pd.datetime.today()))

logger.info("Training finished")

# Save as csv
if not os.path.isdir("results/{}".format(DATASET_NAME)):
    os.makedirs("results/{}".format(DATASET_NAME))
df_libsvm.to_csv("results/{}/csvm.csv".format(DATASET_NAME), index=False)
