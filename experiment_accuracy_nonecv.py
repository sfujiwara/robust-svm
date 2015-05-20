import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import time
import pandas as pd

from src import ersvmdca, rampsvm, enusvm, ersvmutil, ersvmh
#from src_old import ersvm

if __name__ == '__main__':
    ## Set seed
    np.random.seed(0)

    ## Read data set
    name_dataset = 'liver'
    ## filename = 'datasets/LIBSVM/cod-rna/cod-rna.csv'
    ## filename = 'datasets/LIBSVM/heart/heart_scale.csv'
    filename = 'datasets/LIBSVM/liver-disorders/liver-disorders_scale.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape
    num_tr = 138
    num_val = 103
    num_t = 104
    trial = 2
    
    ## Scaling
    ## ersvmutil.libsvm_scale(x)
    ersvmutil.standard_scale(x)

    ## Initial point generated at random
    initial_weight = np.random.normal(size=dim)
    initial_weight = initial_weight / np.linalg.norm(initial_weight)

    ## Candidates of hyper-parameters
    nu_cand = np.linspace(0.75, 0.1, 9)
    cost_cand = np.array([5.**i for i in range(4, -5, -1)])
    ol_ratio = np.array([0., 0.05])
    mu_cand = np.array([0.01, 0.05, 0.1, 0.2])
    s_cand = np.array([-1, 0., 0.5])

    ## Class instances
    ersvm = ersvmdca.LinearPrimalERSVM()
    ersvm.set_initial_point(initial_weight, 0)
    ramp = rampsvm.RampSVM()
    enu = enusvm.EnuSVM()
    var = ersvmh.HeuristicLinearERSVM()
    libsvm = svm.SVC(C=1e0, kernel='linear')

    ## DataFrame for results
    df_dca = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'mu', 'val-acc', 'var-f', 'test-acc', 'test-f', 'VaR', 'tr-CVaR'])
    df_var = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'val-acc', 'val-f', 'test-acc', 'test-f'])
    df_enu = pd.DataFrame(columns=['ratio', 'trial', 'nu', 'val-acc', 'val-f', 'test-acc', 'test-f'])
    df_libsvm = pd.DataFrame(columns=['ratio', 'trial', 'C', 'val-acc', 'val-f', 'test-acc', 'test-f'])
    df_ramp = pd.DataFrame(columns=['ratio', 'trial', 'C', 's', 'val-acc', 'val-f', 'test-acc', 'test-f'])

    ## Loop for outlier ratio
    for i in range(len(ol_ratio)):
        num_ol_tr = int(num_tr * ol_ratio[i])
        num_ol_val = int(num_val * ol_ratio[i])
    ## Loop for random splitting
        for j in range(trial):
            ## Split indices to training, validation, and test set
            ind_rand = np.random.permutation(range(num))
            ind_tr = ind_rand[:num_tr]
            ind_val = ind_rand[num_tr:(num_tr+num_val)]
            ind_t = ind_rand[(num_tr+num_val):]
            ## Copy training and validation set since they will be contaminated
            x_tr = np.array(x[ind_tr])
            y_tr = np.array(y[ind_tr])
            x_val = np.array(x[ind_val])
            y_val = np.array(y[ind_val])
            ## Generate synthetic outliers
            if num_ol_tr > 0:
                outliers = ersvmutil.runif_sphere(radius=20, dim=dim, size=num_ol_tr)
                x_tr[:num_ol_tr] = outliers
                outliers = ersvmutil.runif_sphere(radius=20, dim=dim, size=num_ol_val)
                x_val[:num_ol_val] = outliers

            ## Loop for hyper-parameter tuning
            for k in range(len(nu_cand)):
                print 'Start Ramp Loss SVM'
                ramp.set_cost(cost_cand[k])
                ramp.solve_rampsvm(x_tr, y_tr)
                ramp.show_result()
                #acc_ramp[i, cv] = sum((np.dot(x[ind_t], ramp.weight) + ramp.bias) * y[ind_t] > 0) / float(len(y[ind_t]))
                res_ramp = pd.DataFrame([[ol_ratio[i], j, nu_cand[k], -1.,
                                          sum((np.dot(x_val, ramp.weight) + ramp.bias) * y_val > 0) / float(len(y_val)),
                                          sum((np.dot(x[ind_t], ramp.weight) + ramp.bias) * y[ind_t] > 0) / float(len(y[ind_t]))]],
                                        columns=['ratio','trial','C', 's', 'val-acc','test-acc'])
                df_ramp = df_ramp.append(res_ramp)

                print 'Start ER-SVM (DCA)'
                ersvm.set_nu(nu_cand[k])
                ersvm.set_mu(0.05)
                ersvm.solve_ersvm(x_tr, y_tr)
                ersvm.show_result()
                #acc_dca[i, cv] = ersvm.calc_accuracy(x[ind_t], y[ind_t])
                ersvm.set_initial_point(ersvm.weight, 0)
                res_dca = pd.DataFrame([[ol_ratio[i], j, nu_cand[k], 0.05, ersvm.calc_accuracy(x_val, y_val),
                                         ersvm.calc_accuracy(x[ind_t], y[ind_t]), ersvm.alpha, ersvm.obj[-1]]],
                                       columns=['ratio','trial','nu','mu','val-acc','test-acc','VaR','tr-CVaR'])
                df_dca = df_dca.append(res_dca)

                print 'Start Enu-SVM'
                enu.set_initial_weight(initial_weight)
                enu.set_nu(nu_cand[k])
                enu.solve_enusvm(x_tr, y_tr)
                enu.show_result()
                res_enu = pd.DataFrame([[ol_ratio[i], j, nu_cand[k], enu.calc_accuracy(x_val, y_val),
                                         enu.calc_accuracy(x[ind_t], y[ind_t])]],
                                       columns=['ratio','trial','nu','val-acc','test-acc'])
                df_enu = df_enu.append(res_enu)

                print 'Start ER-SVM (Heuristics)'
                var.set_initial_weight(initial_weight)
                var.set_nu(nu_cand[k])
                var.solve_varmin(x_tr, y_tr)
                var.show_result()
                res_var = pd.DataFrame([[ol_ratio[i], j, nu_cand[k], var.calc_accuracy(x_val, y_val),
                                         var.calc_accuracy(x[ind_t], y[ind_t])]],
                                       columns=['ratio','trial','nu','val-acc','test-acc'])
                df_var = df_var.append(res_var)

                print 'Start LIBSVM'
                start = time.time()
                libsvm.set_params(**{'C':cost_cand[k]})
                libsvm.fit(x_tr, y_tr)
                end = time.time()
                print 'End LIBSVM'
                print 'time:', end - start
                res_libsvm = pd.DataFrame([[ol_ratio[i], j, cost_cand[k], libsvm.score(x_val,y_val), libsvm.score(x[ind_t],y[ind_t])]],
                                       columns=['ratio','trial','C','val-acc','test-acc'])
                df_libsvm = df_libsvm.append(res_libsvm)
