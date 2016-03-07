# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn import svm
from fsvm import ersvmdca, ersvmh, enusvm, rampsvm, svmutil

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
num_tr = 100   # size of training set
num_val = 100  # size of validation set
num_t = 100    # size of test set
radius = 75    # level of outlier
trial = 1

# Candidates of hyper-parameters
nu_cand = np.linspace(0.9, 0.1, 9)
cost_cand = np.array([5.**i for i in range(4, -5, -1)])
ol_ratio = np.array([0., 0.03, 0.05, 0.1])
mu_cand = np.array([0.05])
s_cand = np.array([-1])

# Scaling
# ersvmutil.libsvm_scale(x)
svmutil.standard_scale(x)
# Initial point generated at random
initial_weight = np.random.normal(size=dim)
initial_weight = initial_weight / np.linalg.norm(initial_weight)
# Class instances
ersvm = ersvmdca.LinearPrimalERSVM()
ersvm.set_initial_point(np.array(initial_weight), 0)
var = ersvmh.HeuristicLinearERSVM()
conv_ersvm = ersvmdca.LinearPrimalERSVM()
conv_ersvm.set_initial_point(np.array(initial_weight), 0)
conv_ersvm.set_constant_t(0)
# DataFrame for results
df_dca = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'mu', 'val-acc', 'val-f', 'test-acc', 'test-f', 'VaR', 'tr-CVaR', 'comp_time'])
df_var = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'val-acc', 'val-f', 'test-acc', 'test-f', 'is_convex', 'comp_time'])
df_enusvm = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'val-acc', 'val-f', 'test-acc', 'test-f', 'is_convex', 'comp_time'])
df_libsvm = pd.DataFrame(columns=['ratio', 'trial', 'C', 'val-acc', 'val-f', 'test-acc', 'test-f', 'comp_time'])
df_ramp = pd.DataFrame(columns=['ratio', 'trial', 'C', 's', 'val-acc', 'val-f', 'test-acc', 'test-f', 'comp_time', 'timeout'])
df_conv = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'mu', 'val-acc', 'val-f', 'test-acc', 'test-f', 'VaR', 'tr-CVaR', 'comp_time'])

# Loop for outlier ratio
for i in range(len(ol_ratio)):
    num_ol_tr = int(num_tr * ol_ratio[i])
    num_ol_val = int(num_val * ol_ratio[i])
    # Loop for random splitting
    for j in range(trial):
        # Split indices to training, validation, and test set
        ind_rand = np.random.permutation(range(num))
        ind_tr = ind_rand[:num_tr]
        ind_val = ind_rand[num_tr:(num_tr+num_val)]
        ind_t = ind_rand[(num_tr+num_val):]
        # Copy training and validation set since they will be contaminated
        x_tr = np.array(x[ind_tr])
        y_tr = np.array(y[ind_tr])
        x_val = np.array(x[ind_val])
        y_val = np.array(y[ind_val])
        # Generate synthetic outliers
        if num_ol_tr > 0:
            outliers = svmutil.runif_sphere(radius=radius, dim=dim, size=num_ol_tr)
            x_tr[:num_ol_tr] = outliers
            outliers = svmutil.runif_sphere(radius=radius, dim=dim, size=num_ol_val)
            x_val[:num_ol_val] = outliers

        # Initial point generated at random
        initial_weight = np.random.normal(size=dim)
        initial_weight /= np.linalg.norm(initial_weight)

        # Loop for hyper-parameter tuning
        for k in range(len(nu_cand)):
            # Hyper-parameter s of Ramp SVM
            for l in range(len(mu_cand)):
                print 'Start Ramp Loss SVM'
                model_ramp = rampsvm.RampSVM(C=cost_cand[k])
                model_ramp.cplex_method = 1
                model_ramp.fit(x_tr, y_tr)
                model_ramp.show_result()
                row_ramp = {
                    'ratio': ol_ratio[i],
                    'trial': j,
                    'C': cost_cand[k],
                    's': s_cand[l],
                    'val-acc': model_ramp.score(x_val, y_val),
                    'val-f': model_ramp.f_score(x_val, y_val),
                    'test-acc': model_ramp.score(x[ind_t], y[ind_t]),
                    'test-f': model_ramp.f_score(x[ind_t], y[ind_t]),
                    'comp_time': model_ramp.comp_time,
                    'timeout': model_ramp.timeout
                }
                df_ramp = df_ramp.append(pd.Series(row_ramp, name=pd.datetime.today()))

            # Hyper-parameter mu of ER-SVM
            for l in range(len(mu_cand)):
                print 'Start ER-SVM (DCA)'
                print '(ratio, trial):', (ol_ratio[i], j)
                if nu_cand[k] > mu_cand[l]:
                    ersvm.set_nu(nu_cand[k])
                    ersvm.set_mu(mu_cand[l])
                    ersvm.solve_ersvm(x_tr, y_tr)
                    ersvm.show_result()
                    ersvm.set_initial_point(np.array(initial_weight), 0)
                    row_dca = {
                        'ratio': ol_ratio[i],
                        'trial': j,
                        'nu': nu_cand[k],
                        'mu': mu_cand[l],
                        'val-acc': ersvm.calc_accuracy(x_val, y_val),
                        'val-f': ersvm.calc_f(x_val, y_val),
                        'test-acc': ersvm.calc_accuracy(x[ind_t], y[ind_t]),
                        'test-f': ersvm.calc_f(x[ind_t], y[ind_t]),
                        'VaR': ersvm.alpha,
                        'tr-CVaR': ersvm.obj[-1],
                        'comp_time': ersvm.comp_time
                    }
                    df_dca = df_dca.append(pd.Series(row_dca, name=pd.datetime.today()))

            # Hyper-parameter mu of ER-SVM with t = 0
            for l in range(len(mu_cand)):
                print 'Start ER-SVM (DCA) with t = 0'
                if nu_cand[k] > mu_cand[l]:
                    conv_ersvm.set_nu(nu_cand[k])
                    conv_ersvm.set_mu(mu_cand[l])
                    conv_ersvm.solve_ersvm(x_tr, y_tr)
                    conv_ersvm.show_result()
                    conv_ersvm.set_initial_point(np.array(initial_weight), 0)
                    row_conv = {
                        'ratio': ol_ratio[i],
                        'trial': j,
                        'nu': nu_cand[k],
                        'mu': mu_cand[l],
                        'val-acc': conv_ersvm.calc_accuracy(x_val, y_val),
                        'val-f': conv_ersvm.calc_f(x_val,y_val),
                        'test-acc': conv_ersvm.calc_accuracy(x[ind_t], y[ind_t]),
                        'test-f': conv_ersvm.calc_f(x[ind_t], y[ind_t]),
                        'VaR': conv_ersvm.alpha,
                        'tr-CVaR': conv_ersvm.obj[-1],
                        'comp_time': conv_ersvm.comp_time
                    }
                    df_conv = df_conv.append(pd.Series(row_conv, name=pd.datetime.today()))

            # Enu-SVM
            print 'Start Enu-SVM'
            model_enusvm = enusvm.EnuSVM(nu=nu_cand[k])
            model_enusvm.fit(x_tr, y_tr, initial_weight)
            row_enusvm = {
                'ratio': ol_ratio[i],
                'trial': j,
                'nu': nu_cand[k],
                'val-acc': model_enusvm.score(x_val, y_val),
                'val-f': model_enusvm.f1_score(x_val, y_val),
                'test-acc': model_enusvm.score(x[ind_t], y[ind_t]),
                'test-f': model_enusvm.f1_score(x[ind_t], y[ind_t]),
                'is_convex': model_enusvm.convexity,
                'comp_time': model_enusvm.comp_time
            }
            df_enusvm = df_enusvm.append(pd.Series(row_enusvm, name=pd.datetime.today()))

            print 'Start ER-SVM (Heuristics)'
            var.set_initial_weight(np.array(initial_weight))
            var.set_nu(nu_cand[k])
            var.set_gamma(0.03/nu_cand[k])
            var.solve_varmin(x_tr, y_tr)
            var.show_result()
            row_var = {
                'ratio': ol_ratio[i],
                'trial': j,
                'nu': nu_cand[k],
                'val-acc': var.score(x_val, y_val),
                'val-f': var.f_score(x_val, y_val),
                'test-acc': var.score(x[ind_t], y[ind_t]),
                'test-f': var.f_score(x[ind_t], y[ind_t]),
                'is_convex': var.is_convex,
                'comp_time': var.comp_time
            }
            df_var = df_var.append(pd.Series(row_var, name=pd.datetime.today()))

            # C-SVM (LIBSVM)
            print 'Start LIBSVM'
            start = time.time()
            model_libsvm = svm.SVC(C=cost_cand[k], kernel='linear', max_iter=-1)
            model_libsvm.fit(x_tr, y_tr)
            end = time.time()
            print 'End LIBSVM'
            print 'time:', end - start
            row_libsvm = {
                'ratio': ol_ratio[i],
                'trial': j,
                'C': cost_cand[k],
                'val-acc': model_libsvm.score(x_val, y_val),
                'val-f': f1_score(y_val, model_libsvm.predict(x_val)),
                'test-acc': model_libsvm.score(x[ind_t], y[ind_t]),
                'test-f': f1_score(y[ind_t], model_libsvm.predict(x[ind_t]))
            }
            df_libsvm = df_libsvm.append(pd.Series(row_libsvm, name=pd.datetime.today()))

#pd.set_option('line_width', 200)

# Save as csv
# df_dca.to_csv(dir_name_result+'dca.csv', index=False)
# df_enu.to_csv(dir_name_result+'enu.csv', index=False)
# df_var.to_csv(dir_name_result+'var.csv', index=False)
# df_ramp.to_csv(dir_name_result+'ramp.csv', index=False)
# df_libsvm.to_csv(dir_name_result+'libsvm.csv', index=False)
# df_conv.to_csv(dir_name_result+'conv.csv', index=False)
