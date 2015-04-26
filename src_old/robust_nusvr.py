## -*- conding: utf-8 -*-

import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # Ubuntu
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import cplex
import time
from sklearn.metrics import pairwise_kernels

## Robust Kernel nu-SVR
def robust_nusvr(x, y, cost, nu, mu, kernel, gamma, with_bias=True):
    print '===== (C, nu, mu, kernel):', (cost, nu, mu, kernel), '====='
    ##### Constant values #####
    num, dim = x.shape
    max_itr = 50
    ##### Initial point #####
    eta = np.ones(num)
    ##### Compute kernel gram matrix #####
    if kernel == 'linear': kmat = pairwise_kernels(x, metric='linear')
    elif kernel == 'rbf': kmat = pairwise_kernels(x, x, metric='rbf', gamma=gamma)
    elif kernel == 'polynomial': kmat = pairwise_kernels(x, metric='polynomial', coef0=coef0, degree=degree)
    else: print 'Undefined Kernel!!'
    ## kmat = np.round(kmat, 10)
    kmat = np.round(kmat + np.eye(num) * 1e-8, 10)
    ##### Cplex object #####
    c = cplex.Cplex()
    c.set_results_stream(None)
    c.parameters.qpmethod.set(2)
    ##### Names of variables #####
    names_a = ['a%s' % i for i in range(num)]
    names_xi = ['xi%s' % i for i in range(num)]
    ##### Set variables and linear objective #####
    c.variables.add(names=names_a, lb=[-cplex.infinity]*num)
    if with_bias:
        c.variables.add(names=['b'], lb=[-cplex.infinity])
    c.variables.add(names=names_xi, obj=[cost/num]*num)
    ## c.variables.add(names=['rho'], lb=[-cplex.infinity], obj=[cost*(nu+mu)])
    c.variables.add(names=['rho'], obj=[cost*(nu+mu)])
    ##### Set linear constraints #####
    ## K alf + b - rho - xi <= y
    if with_bias:
        linexpr = [[names_a+['b', 'rho', 'xi%s' % j], list(kmat[j])+[1., -1., -1.]] for j in range(num)]
    else:
        linexpr = [[names_a+['rho', 'xi%s' % j], list(kmat[j])+[-1., -1.]] for j in range(num)]
    c.linear_constraints.add(lin_expr=linexpr, senses='L'*num, rhs=list(y))
    ## K alf + b + rho + xi >= y
    if with_bias:
        linexpr = [[names_a+['b', 'rho', 'xi%s' % j], list(kmat[j])+[1., 1., 1.]] for j in range(num)]
    else:
        linexpr = [[names_a+['rho', 'xi%s' % j], list(kmat[j])+[1., 1.]] for j in range(num)]
    c.linear_constraints.add(lin_expr=linexpr, senses='G'*num, rhs=list(y))
    ##### Set quadratic objective #####
    quadexpr = [[names_a, list(kmat[i])] for i in range(num)]
    quadexpr = quadexpr + [[[],[]]] * (num+2)
    c.objective.set_quadratic(quadexpr)
    ##### Iteration #####
    for i in range(max_itr):
        ##### Solve subproblem #####
        c.solve()
        alf = np.array(c.solution.get_values(names_a))
        if with_bias:
            b = c.solution.get_values('b')
        else:
            b = 0.
        rho = c.solution.get_values('rho')
        xi = np.array(c.solution.get_values(names_xi))
        ##### Decision values #####
        dv = np.dot(kmat, alf) + b
        ##### sign(K alf + b - y) #####
        sig_err = np.sign(dv-y)
        ##### Update r_i(alf, b) #####
        risk = np.abs(dv-y)
        ##### Update eta #####
        ind_sorted = np.argsort(risk)
        eta_bef = eta
        eta = np.zeros(num)
        eta[ind_sorted[range(int(np.ceil(num*(1-mu))))]] = 1.
        ##### Termination #####
        if np.all(eta == eta_bef):
            print 'CONVERGED'
            break
        ##### Update linear objective #####
        beta = sig_err * (1-eta)
        tmp = (kmat.T * beta).T
        tmp = -np.sum(tmp, axis=0) * cost / num
        c.objective.set_linear(zip(names_a, list(tmp)))
        if with_bias:
            c.objective.set_linear([['b', -sum(sig_err * (1-eta))*cost/num]])
    print 'Total Iteration:', i+1
    print 'Objective Value (subproblem):', c.solution.get_objective_value()
    print 'Primal Feasible:', c.solution.is_primal_feasible()
    print 'Status:', c.solution.get_status_string()
    print '\n'
    return c, eta, kmat

## Robust One-class SVM
def robust_ocsvm(x, nu, mu, kernel, gamma):
    print '===== (C, nu, mu, kernel):', (cost, nu, mu, kernel), '====='
    ##### Constant values #####
    num, dim = x.shape
    max_itr = 50
    ##### Initial point #####
    eta = np.ones(num)
    ##### Compute kernel gram matrix #####
    if kernel == 'linear': kmat = pairwise_kernels(x, metric='linear')
    elif kernel == 'rbf': kmat = pairwise_kernels(x, x, metric='rbf', gamma=gamma)
    elif kernel == 'polynomial': kmat = pairwise_kernels(x, metric='polynomial', coef0=coef0, degree=degree)
    else: print 'Undefined Kernel!!'
    ## kmat = np.round(kmat, 10)
    kmat = kmat + np.eye(num) * 1e-8
    ##### Cplex object #####
    c = cplex.Cplex()
    c.set_results_stream(None)
    c.parameters.qpmethod.set(2)
    ##### Names of variables #####
    names_a = ['a%s' % i for i in range(num)]
    names_xi = ['xi%s' % i for i in range(num)]
    ##### Set variables and linear objective #####
    c.variables.add(names=names_a, lb=[-cplex.infinity]*num)
    if with_bias:
        c.variables.add(names=['b'], lb=[-cplex.infinity])
    c.variables.add(names=names_xi, obj=[cost/num]*num)
    ## c.variables.add(names=['rho'], lb=[-cplex.infinity], obj=[cost*(nu+mu)])
    c.variables.add(names=['rho'], obj=[cost*(nu+mu)])
    ##### Set linear constraints #####
    ## K alf + b - rho - xi <= y
    if with_bias:
        linexpr = [[names_a+['b', 'rho', 'xi%s' % j], list(kmat[j])+[1., -1., -1.]] for j in range(num)]
    else:
        linexpr = [[names_a+['rho', 'xi%s' % j], list(kmat[j])+[-1., -1.]] for j in range(num)]
    c.linear_constraints.add(lin_expr=linexpr, senses='L'*num, rhs=list(y))
    ## K alf + b + rho + xi >= y
    if with_bias:
        linexpr = [[names_a+['b', 'rho', 'xi%s' % j], list(kmat[j])+[1., 1., 1.]] for j in range(num)]
    else:
        linexpr = [[names_a+['rho', 'xi%s' % j], list(kmat[j])+[1., 1.]] for j in range(num)]
    c.linear_constraints.add(lin_expr=linexpr, senses='G'*num, rhs=list(y))
    ##### Set quadratic objective #####
    quadexpr = [[names_a, list(kmat[i])] for i in range(num)]
    quadexpr = quadexpr + [[[],[]]] * (num+2)
    c.objective.set_quadratic(quadexpr)
    ##### Iteration #####
    for i in range(max_itr):
        ##### Solve subproblem #####
        c.solve()
        alf = np.array(c.solution.get_values(names_a))
        if with_bias:
            b = c.solution.get_values('b')
        else:
            b = 0.
        rho = c.solution.get_values('rho')
        xi = np.array(c.solution.get_values(names_xi))
        ##### Decision values #####
        dv = np.dot(kmat, alf) + b
        ##### sign(K alf + b - y) #####
        sig_err = np.sign(dv-y)
        ##### Update r_i(alf, b) #####
        risk = np.abs(dv-y)
        ##### Update eta #####
        ind_sorted = np.argsort(risk)
        eta_bef = eta
        eta = np.zeros(num)
        eta[ind_sorted[range(int(np.ceil(num*(1-mu))))]] = 1.
        ##### Termination #####
        if np.all(eta == eta_bef):
            print 'CONVERGED'
            break
        ##### Update linear objective #####
        beta = sig_err * (1-eta)
        tmp = (kmat.T * beta).T
        tmp = -np.sum(tmp, axis=0) * cost / num
        c.objective.set_linear(zip(names_a, list(tmp)))
        if with_bias:
            c.objective.set_linear([['b', -sum(sig_err * (1-eta))*cost/num]])
    print 'Total Iteration:', i+1
    print 'Objective Value (subproblem):', c.solution.get_objective_value()
    print 'Primal Feasible:', c.solution.is_primal_feasible()
    print 'Status:', c.solution.get_status_string()
    print '\n'
    return c, eta, kmat

if __name__ == '__main__':
    ##### Read data set from csv #####
    dataset = np.loadtxt('Dataset/LIBSVM/housing/housing_scale.csv', delimiter=',')
    x = dataset[:, 1:]
    y = dataset[:, 0]
    num, dim = x.shape
    ##### Synthetic data set #####
    ## mu = [1, 1]
    ## cov = [[1.1, -1], [-1, 1.1]]
    ## num = 500
    ## np.random.seed(0)
    ## x, y = np.random.multivariate_normal(mu, cov, num).T
    ## x = x[:, np.newaxis]
    ## plt.plot(x, y, 'x')
    ## plt.show()
    ##### Split training set and test set #####
    np.random.seed(0)
    num_train = 200
    ind_train = np.sort(np.random.choice(range(num), num_train, replace=False))
    ind_test = np.array(list(set(range(num)).difference(ind_train)))
    ##### Names of variables #####
    names_a = ['a%s' % i for i in range(num_train)]
    names_xi = ['xi%s' % i for i in range(num_train)]
    ##### Hyper-parameters #####
    cost = 1e5
    dist = []
    for i in range(200):
        for j in range(i+1, 200):
            dist.append(np.linalg.norm(x[i] - x[j]))
    dist = np.array(dist)
    gamma = 1. / np.median(dist)**2
    ## nu_cand = np.arange(0.05, 1.0, 0.05)
    ## mu_cand = np.arange(0.0, 0.65, 0.05)
    ## trial = 30
    nu_cand = np.array([0.1, 0.3, 0.5, 0.7])
    mu_cand = np.array([0.1, 0.2])
    ## nu_cand = np.arange(0.05, 1.0, 0.05)
    ## mu_cand = np.arange(0.0, 1.0, 0.05)
    trial = 30
    kappa_cand = np.arange(0, 130, 10)
    ##### Kernel function #####
    kernel = 'linear'
    ##### Bias #####
    with_bias = False
    ##### Test run #####
    df_result = pd.DataFrame(columns=['mu', 'nu', 'kappa', 'max_norm', 'max_b', 'max_rmse', 'max_obj'])
    ## Loop for mu
    for i in range(len(mu_cand)):
        mu = mu_cand[i]
        ## Loop for nu
        for j in range(len(nu_cand)):
            nu = nu_cand[j]
            ## Loop for kappa
            for k in range(len(kappa_cand)):
                num_ol = kappa_cand[k]
                max_norm = -1e10
                max_b = -1e10
                max_rmse = -1e10
                for seed in range(trial):
                    x_train = np.array(x[ind_train])
                    y_train = np.array(y[ind_train])
                    ## Generate noise
                    np.random.seed(seed)
                    ind_ol = np.random.choice(range(num_train), num_ol, replace=False)
                    ## y_train[ind_ol] += np.random.normal(0, 30, num_ol)
                    y_train[ind_ol] += np.random.normal(0, 100, num_ol)
                    ## y_train[ind_ol] += np.random.normal(0, 500, num_ol)
                    if nu + mu < 1.:
                        ## Solve max-type nu-SVR
                        res, eta, kmat = robust_nusvr(x_train, y_train, cost, nu, mu, kernel, gamma, with_bias)
                        alf = np.array(res.solution.get_values(names_a))
                        if with_bias:
                            b = res.solution.get_values('b')
                        else: b = 0.
                        rho = res.solution.get_values('rho')
                        xi = np.array(res.solution.get_values(names_xi))
                        dv = np.dot(kmat, alf) + b
                        risk = np.abs(dv - y_train)
                        dv_test = np.dot(pairwise_kernels(x[ind_test], x_train, metric=kernel), alf) + b
                        rmse = np.sqrt(np.mean((dv_test - y[ind_test])**2))
                        norm = np.sqrt(np.dot(alf, np.dot(kmat, alf)))
                        obj = np.dot(alf, np.dot(kmat, alf)) / (2*cost*nu) + np.dot(eta, xi) / (nu*num) + rho
                        max_norm = np.max([max_norm, norm])
                        max_b = np.max([max_b, b])
                        max_rmse = np.max([max_rmse, rmse])
                df_result = df_result.append(pd.Series([mu, nu, num_ol, max_norm, max_b, max_rmse], index=['mu', 'nu', 'kappa', 'max_norm', 'max_b', 'max_rmse']), ignore_index=True)

    tmp = 13 * 0
    tmp2 = 'max_norm'
    plt.plot( df_result['kappa'][0:12], df_result[tmp2][tmp:(tmp+12)], label='nu = 0.1')
    tmp = 13 * 1
    plt.plot( df_result['kappa'][0:12], df_result[tmp2][tmp:(tmp+12)], label='nu = 0.3')
    tmp = 13 * 2
    plt.plot( df_result['kappa'][0:12], df_result[tmp2][tmp:(tmp+12)], label='nu = 0.5')
    tmp = 13 * 3
    plt.plot( df_result['kappa'][0:12], df_result[tmp2][tmp:(tmp+12)], label='nu = 0.7')
    plt.axvline(x=20, color='black')
    plt.grid()
    plt.xlabel('kappa')
    plt.ylabel(tmp2)
    plt.legend()
    plt.show()

    ## Write csv file
    ## df_result.to_csv('df_result.csv', index=False)

    ## Old
    ## tab_bias = np.zeros([trial, len(mu_cand), len(nu_cand)])
    ## tab_norm = np.zeros([trial, len(mu_cand), len(nu_cand)])
    ## tab_rmse = np.zeros([trial, len(mu_cand), len(nu_cand)])
    ## tab_obj = np.zeros([trial, len(mu_cand), len(nu_cand)])
    ## tab_rho = np.zeros([trial, len(mu_cand), len(nu_cand)])
    ## df_res = pd.DataFrame(columns=['mu', 'nu', 'norm', 'bias', 'rmse', 'obj'])
    ## for i in range(trial):
    ##     x_train = x[ind_train]
    ##     y_train = y[ind_train]
    ##     num_ol = 20
    ##     ind_ol = np.random.choice(range(num_train), num_ol, replace=False)
    ##     y_train[ind_ol] += np.random.normal(0, 100, num_ol)
    ##     ## x_train[ind_ol] += np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim)*100, size=num_ol)
    ##     for j in range(len(mu_cand)):
    ##         mu = mu_cand[j]
    ##         for k in range(len(nu_cand)):
    ##             nu = nu_cand[k]
    ##             if nu + mu < 1.:
    ##             ## if True:
    ##                 res, eta, kmat = robust_kernel_nusvr(x_train, y_train, cost, nu, mu, kernel, gamma)
    ##                 alf = np.array(res.solution.get_values(names_a))
    ##                 b = res.solution.get_values('b')
    ##                 rho = res.solution.get_values('rho')
    ##                 xi = np.array(res.solution.get_values(names_xi))
    ##                 ## w = np.dot(x_train.T, alf)
    ##                 dv = np.dot(kmat, alf) + b
    ##                 risk = np.abs(dv - y_train)
    ##                 ## ind_ol = np.where(eta==0)[0]
    ##                 ## rmse = np.sqrt(np.mean((np.dot(x[ind_test], w) + b - y[ind_test])**2))
    ##                 dv_test = np.dot(pairwise_kernels(x[ind_test], x_train, metric=kernel), alf) + b
    ##                 rmse = np.sqrt(np.mean((dv_test - y[ind_test])**2))
    ##                 norm = np.sqrt(np.dot(alf, np.dot(kmat, alf)))
    ##                 obj = np.dot(alf, np.dot(kmat, alf)) / (2*cost*nu) + np.dot(eta, xi) / (nu*num) + rho
    ##                 tab_rmse[i, j, k] = rmse
    ##                 tab_bias[i, j, k] = b
    ##                 ## tab_norm[i, j, k] = np.linalg.norm(w)
    ##                 tab_norm[i, j, k] = norm
    ##                 tab_rho[i, j, k] = rho
    ##                 tab_obj[i, j, k] = obj
    ##                 ## tab_obj[i, j, k] = np.linalg.norm(w)**2 / (2*cost*nu) + np.dot(eta, xi) / (nu*num) + rho
                    
    ## tab = []
    ## norm_max = np.amax(tab_norm, axis=0)
    ## bias_max = np.amax(tab_bias, axis=0)
    ## rmse_max = np.amax(tab_rmse, axis=0)
    ## obj_max = np.amax(tab_obj, axis=0)
    ## for i in range(len(mu_cand)):
    ##     for j in range(len(nu_cand)):
    ##         if mu_cand[i] + nu_cand[j] < 1:
    ##             tab.append([mu_cand[i], nu_cand[j], norm_max[i,j], bias_max[i,j], rmse_max[i,j], obj_max[i,j]])
    ## tab = np.array(tab)

    ## ##### 3D plot #####
    ## fig = plt.figure()
    ## ax = Axes3D(fig)
    ## nu, mu = np.meshgrid(nu_cand, mu_cand)
    ## ## ax.plot_wireframe(mu, nu, np.amax(tab_bias, axis=0), label='bias')
    ## ## ax.plot_wireframe(mu, nu, np.amax(tab_rmse, axis=0), label='RMSE')
    ## ## ax.plot_wireframe(mu, nu, np.amax(tab_norm, axis=0), label='norm')
    ## ## ax.plot_wireframe(mu, nu, np.amax(tab_obj, axis=0), label='objective value')
    ## ax.scatter3D(tab[:,0], tab[:,1], tab[:,2], label='norm')
    ## ## ax.plot_wireframe(mu, nu, np.amax(tab_rho, axis=0), label='rho')
    ## ## ax.scatter3D(mu, nu, np.amax(tab_norm, axis=0))
    ## plt.xlabel('mu')
    ## plt.ylabel('nu')
    ## plt.legend()
    ## plt.show()

    ## ##### Save csv files #####
    ## ## np.savetxt('bias.csv', np.amax(tab_bias, axis=0), fmt='%0.9f', delimiter=',')
    ## ## np.savetxt('obj.csv', np.amax(tab_obj, axis=0), fmt='%0.9f', delimiter=',')
    ## ## np.savetxt('norm.csv', np.amax(tab_norm, axis=0), fmt='%0.9f', delimiter=',')
    ## ## np.savetxt('rmse.csv', np.amax(tab_rmse, axis=0), fmt='%0.9f', delimiter=',')
    ## np.savetxt('test.csv', np.array(tab), fmt='%0.9f', delimiter=',')

    ## ## res, eta, kmat = robust_kernel_nusvr(x_train, y_train, cost, 0.7, 0.1, kernel)
    ## ## alf = np.array(res.solution.get_values(names_a))
    ## ## b = res.solution.get_values('b')
    ## ## rho = res.solution.get_values('rho')
    ## ## xi = np.array(res.solution.get_values(names_xi))
    ## ## w = np.dot(x_train.T, alf)
    ## ## isfeasible = np.abs(np.dot(kmat, alf) + b - y_train) <= rho + xi + 1e-8
