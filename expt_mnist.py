# -*- coding: utf-8 -*-

# import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
# from sklearn.metrics import f1_score
# from sklearn import svm

# Import my modules
from fsvm import ersvm, ersvmh, enusvm, rampsvm, svmutil

# Load MNIST data
mnist = fetch_mldata('MNIST original', data_home='data/sklearn')
mnist1 = mnist.data[mnist.target == 1]  # size = 7877
mnist7 = mnist.data[mnist.target == 7]  # size = 7293
x = np.vstack([mnist1, mnist7]).astype(float)
y = np.array([1] * len(mnist1) + [-1] * len(mnist7))
num, dim = x.shape

# Set seed
np.random.seed(0)

# Experimental set up
num_tr = 6068   # size of training set
num_val = 4551  # size of validation set
num_t = 4551    # size of test set
radius = 75     # level of outlier
trial = 1

# Candidates of hyper-parameters
nu_list = np.linspace(0.9, 0.1, 9)
c_list = np.array([5. ** i for i in range(4, -5, -1)])
outlier_ratio = np.array([0., 0.03, 0.05, 0.1])

# Scaling
# ersvmutil.libsvm_scale(x)
svmutil.standard_scale(x)

# Initial point generated at random
initial_weight = np.random.normal(size=dim)
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
            x_tr[:num_ol_tr] = svmutil.runif_sphere(radius=radius, dim=dim, size=num_ol_tr)
            x_val[:num_ol_val] = svmutil.runif_sphere(radius=radius, dim=dim, size=num_ol_val)
        # Initial point generated at random
        initial_weight = np.random.normal(size=dim)
        initial_weight /= np.linalg.norm(initial_weight)
        # Loop for hyper parameters tuning
        for k in range(len(nu_list)):
            # Ramp Loss SVM
            print 'Train ramp loss SVM (C, ratio, trial):', (c_list[k], outlier_ratio[i], j)
            model_ramp = rampsvm.RampSVM(C=c_list[k])
            model_ramp.fit(x_tr, y_tr)
            print 'time:', model_ramp.comp_time, '\n'
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
            print 'Train ER-SVM with DCA (nu, ratio, trial):', (nu_list[k], outlier_ratio[i], j)
            model_ersvm = ersvm.LinearERSVM(nu=nu_list[k])
            model_ersvm.fit(x_tr, y_tr, initial_weight)
            print 'time:', model_ersvm.comp_time, '\n'
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
            print 'Train Enu-SVM (nu, ratio, trial):', (nu_list[k], outlier_ratio[i], j)
            model_enusvm = enusvm.EnuSVM(nu=nu_list[k])
            model_enusvm.fit(x_tr, y_tr, initial_weight)
            print 'time:', model_enusvm.comp_time, '\n'
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
            print 'Train ER-SVM with Heuristics (nu, ratio, trial):', (nu_list[k], outlier_ratio[i], j)
            model_var = ersvmh.HeuristicLinearERSVM(nu=nu_list[k], gamma=0.03 / nu_list[k])
            model_var.fit(x_tr, y_tr, initial_weight)
            print 'time:', model_var.comp_time, '\n'
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
            # C-SVM (LIBSVM)
            # print 'Start LIBSVM'
            # start = time.time()
            # model_libsvm = svm.SVC(C=c_list[k], kernel='linear', max_iter=-1)
            # model_libsvm.fit(x_tr, y_tr)
            # end = time.time()
            # print 'End LIBSVM'
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

# pd.set_option('line_width', 200)

# Save as csv
# df_dca.to_csv(dir_name_result+'dca.csv', index=False)
# df_enu.to_csv(dir_name_result+'enu.csv', index=False)
# df_var.to_csv(dir_name_result+'var.csv', index=False)
# df_ramp.to_csv(dir_name_result+'ramp.csv', index=False)
# df_libsvm.to_csv(dir_name_result+'libsvm.csv', index=False)
# df_conv.to_csv(dir_name_result+'conv.csv', index=False)
