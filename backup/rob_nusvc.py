# -*- conding: utf-8 -*-

import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # Ubuntu

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cplex
import time
from sklearn.metrics import pairwise_kernels
from sklearn import svm
import pandas as pd
import time

# Robust-nu-SVM using linear kernel
def robust_linear_nusvm(x, y, nu, mu, w, b):
    MAX_ITR = 10
    num, dim = x.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    # Set name of variables
    w_names = ['w%s' % i for i in range(dim)]
    xi_names = ['xi%s' % i for i in range(num)]
    # Set variables and linear objective
    c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
    c.variables.add(names=xi_names, obj=[1./num]*num)
    c.variables.add(names=['b'], lb=[-cplex.infinity])
    c.variables.add(names=['rho'], lb=[-cplex.infinity], obj=[-(nu-mu)])
    # Set quadratic objective: |w|^2 / 2
    c.objective.set_quadratic_coefficients(zip(range(dim),
                                               range(dim), [1]*dim))
    # set linear constraints y(wx + b) + xi - rho >= 0
    for i in xrange(num):
        variables = w_names + ['b'] + ['xi%s' % i] + ['rho']
        coeffs = list(x[i]*y[i]) + [y[i]] + [1] + [-1]
        linexpr = [[variables, coeffs]]
        c.linear_constraints.add(lin_expr=linexpr, senses='G',
                                 rhs=[0], names=['margin%s' % i])
    # Iteration
    for i in range(MAX_ITR):
        # Update risk
        risk = - y * (np.dot(x, w) + b)
        # Update eta
        eta = calc_eta(risk, mu)
        # Update objective
        grad_w = - np.dot(y*(1-eta), x) / (mu*num)
        grad_b = - np.dot(1-eta, y) / (mu*num)
        c.objective.set_linear('b', mu*grad_b)
        c.objective.set_linear(zip(w_names, mu*grad_w))
        # Solve subproblem
        c.solve()
        # update (w, b)
        w_old = w
        w = np.array(c.solution.get_values(w_names))
        b = c.solution.get_values('b')
        rho = c.solution.get_values('rho')
        xi = c.solution.get_values(xi_names)
        #print w, b
        # Stopping criteria
        obj_val = c.solution.get_objective_value()
        obj_val = np.dot(w, w)/2 - (nu-mu)*rho + np.dot(eta, xi)/num
        print np.linalg.norm(w-w_old), obj_val
        if np.linalg.norm(w-w_old) < 1e-5:
            print 'ITERATION:', i
            return c, eta
    print 'OVER MAXIMUM ITERATION'
    return c, eta

# Robust-nu-SVM
def robust_kernel_nusvm(x, y, nu, mu, a, b, kernel):
    MAX_ITR = 10
    num, dim = x.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    # Set name of variables
    a_names = ['a%s' % i for i in range(num)]
    xi_names = ['xi%s' % i for i in range(num)]
    # Compute kernel matrix
    print 'Begin to compute kernel matrix'
    kmat = kernel_matrix(x, kernel)
    kmat = kmat + 1e-10*np.eye(num)
    print 'Done'
    # Set variables and linear objective
    c.variables.add(names = a_names, lb = [-cplex.infinity]*num, ub = [cplex.infinity]*num)
    c.variables.add(names=xi_names, obj=[1./num]*num)
    c.variables.add(names=['b'], lb=[-cplex.infinity])
    c.variables.add(names=['rho'], lb=[-cplex.infinity], obj=[-(nu-mu)])
    # Initialize risk
    risks = - y * (np.dot(kmat, a) + b)
    # Initialize eta
    eta = calc_eta(risks, mu)
    # Set quadratic objective: |w|^2 / 2
    #kmat_sparse = [cplex.SparsePair(ind=a_names, val=list(kmat[i])) for i in range(num)]
    #c.objective.set_quadratic(kmat_sparse)
    print 'Begin to set quadratic objective'
    for i in range(num):
        for j in range(i,num):
            c.objective.set_quadratic_coefficients(i, j, kmat[i,j])
            #c.objective.set_quadratic_coefficients(j, i, kmat[i,j])
    print 'Done'
    # set linear constraints y(wx + b) + xi - rho >= 0
    print 'Begin to set linear constraints'
    for i in xrange(num):
        names_variables = a_names + ['b'] + ['xi%s' % i] + ['rho']
        coeffs = list(y * kmat[i]) + [y[i], 1, -1]
        c.linear_constraints.add(names=['margin%s' % i], senses='G', lin_expr=[[names_variables, coeffs]])
    print 'Done'
    # Iteration
    c.parameters.qpmethod.set(1)
    for i in xrange(MAX_ITR):
        print '\nIteration:\t', i+1
        # Update objective function
        # - Sum_i{1 - eta_i^k} * y_i * b
        print 'Begin to update objective for b'
        c.objective.set_linear('b', np.dot(1-eta, y) / num)
        print 'Finished to update objective'
        print 'Begin to update objective for w' # bottle neck
        #coeffs = np.array([sum((1-eta[k]) * y[k] * kmat[j,k] for k in range(num)) for j in range(num)]) / num
        coeffs1 = np.dot(kmat, y*(1-eta)) / num
        c.objective.set_linear(zip(a_names, coeffs1))
        print 'Finished to update objective'
        # Solve subproblem
        print 'start'
        c.solve()
        print 'end'
        a     = np.array(c.solution.get_values(a_names))
        xi    = np.array(c.solution.get_values(xi_names))
        bias  = c.solution.get_values('b')
        rho = c.solution.get_values('rho')
        # Calculate risk r_i(w_k, b_k)
        risks = - y * (np.dot(kmat, a) + b)
        # Update eta
        eta = calc_eta(risks, mu)
    return c, eta

# Calculate eta
def calc_eta(risks, mu):
    m = len(risks)
    indices_sorted = np.argsort(risks)[::-1]
    eta = np.zeros(m)
    eta[indices_sorted[range(int(np.ceil(m*mu)))]] = 1.
    eta[indices_sorted[int(np.ceil(m*mu)-1)]] -= np.ceil(m*mu) - m*mu
    eta = 1 - eta
    return eta

def robust_nusvm(dmat, y, nu, mu, kernel, eta='ones', gamma=1., coef0=0., degree=2):
    ##### Constant values #####
    EPS = 1e-3
    MAX_ITR = 15
    NUM, DIM = dmat.shape
    ##### Initial point #####
    if eta=='ones':
        eta = np.ones(NUM)
    ##### Compute kernel gram matrix #####
    if kernel == 'linear':
        kmat = pairwise_kernels(x, metric='linear')
    elif kernel == 'rbf':
        kmat = pairwise_kernels(x, x, metric='rbf', gamma=gamma)
    elif kernel == 'polynomial':
        kmat = pairwise_kernels(x, metric='polynomial', coef0=coef0, degree=degree)
    else: print 'Undefined Kernel!!'
    qmat = (kmat.T * y).T * y + 1e-10*np.eye(NUM)
    qmat = np.round(qmat, 10)
    ##### CPLEX object #####
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set variables #####
    c.variables.add(obj=[0]*NUM)
    ##### Set quadratic objective #####
    c.objective.set_quadratic([[range(NUM), list(qmat[i])] for i in range(NUM)])
    ##### Set linear constraint #####
    c.linear_constraints.add(lin_expr=[[range(NUM), list(y)]], senses='E', rhs=[0])
    c.linear_constraints.add(lin_expr=[[range(NUM), [1]*NUM]], senses='E', rhs=[nu-mu])
    ##### Set QP optimization method #####
    c.parameters.qpmethod.set(0)
    for i in xrange(MAX_ITR):
        ##### Update constraints #####
        c.variables.set_lower_bounds(zip(range(NUM), list((eta-1)/NUM)))
        c.variables.set_upper_bounds(zip(range(NUM), list(eta/NUM)))
        ##### Solve subproblem #####
        c.solve()
        print c.solution.status[c.solution.get_status()]
        print 'OBJ_VAL:', c.solution.get_objective_value()
        alpha = np.array(c.solution.get_values())
        print np.round(alpha, 5)
        ##### Compute bias, rho, and decision values #####
        ind_mv_p = [j for j in xrange(NUM) if EPS <= (alpha[j]*NUM) <= 1-EPS and y[j] > 0.9]
        ind_mv_n = [j for j in xrange(NUM) if EPS <= (alpha[j]*NUM) <= 1-EPS and y[j] < -0.9]
        wx = np.dot(alpha*y, kmat)
        print wx[ind_mv_p], wx[ind_mv_n]
        bias = -(np.mean(wx[ind_mv_p])+np.mean(wx[ind_mv_n])) / 2
        rho = (np.mean(wx[ind_mv_p])-np.mean(wx[ind_mv_n])) / 2
        print 'BIAS =', bias
        dv = wx + bias
        ##### Update eta #####
        risks = - dv * y
        eta_new = np.zeros(NUM)
        ind_sorted = np.argsort(risks)[::-1]
        eta_new[ind_sorted[range(int(np.ceil(NUM*mu)))]] = 1.
        eta_new[ind_sorted[int(np.ceil(NUM*mu)-1)]] -= np.ceil(NUM*mu) - NUM*mu
        eta_new = 1 - eta_new
        if all(eta == eta_new):
            print 'CONVERGED: TOTAL_ITR =', i+1
            return c, bias, rho, eta
        else: eta = eta_new
    print 'OVER MAXIMUM ITERATION'
    return c, bias, rho, eta

##### Robust one-class SVM #####
def robust_ocsvm(dmat, nu, mu, kernel, gamma=1.0, coef0=0., degree=2):
    print '===== (nu, mu, kernel):', (nu, mu, kernel), '====='
    ##### Constant values #####
    EPS = 1e-3
    MAX_ITR = 15
    NUM, DIM = dmat.shape
    ##### Initial point #####
    eta = np.ones(NUM)
    ##### Compute kernel gram matrix #####
    if kernel == 'linear':
        kmat = pairwise_kernels(dmat, metric='linear')
    elif kernel == 'rbf':
        kmat = pairwise_kernels(dmat, metric='rbf', gamma=gamma)
    elif kernel == 'polynomial':
        kmat = pairwise_kernels(dmat, metric='polynomial', coef0=coef0, degree=degree)
    else:
        print 'Undefined Kernel!!'
    kmat = np.round(kmat + 1e-8*np.eye(NUM), 10)
    ##### CPLEX object #####
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set variables #####
    c.variables.add(obj=[0]*NUM)
    ##### Set quadratic objective #####
    c.objective.set_quadratic([[range(NUM), list(kmat[i])] for i in range(NUM)])
    ##### Set linear constraint #####
    #c.linear_constraints.add(lin_expr=[[range(NUM), list(y)]], senses='E', rhs=[0])
    c.linear_constraints.add(lin_expr=[[range(NUM), [1]*NUM]], senses='E', rhs=[nu-mu])
    ##### Set QP optimization method #####
    c.parameters.qpmethod.set(1)
    for i in xrange(MAX_ITR):
        ##### Update constraints #####
        c.variables.set_lower_bounds(zip(range(NUM), list((eta-1)/NUM)))
        c.variables.set_upper_bounds(zip(range(NUM), list(eta/NUM)))
        ##### Solve subproblem #####
        c.solve()
        print c.solution.status[c.solution.get_status()]
        print 'OBJ_VAL:', c.solution.get_objective_value()
        alpha = np.array(c.solution.get_values())
        ##### Find SVs and MVs #####
        ind_mv = [j for j in xrange(NUM) if EPS <= (alpha[j]*NUM) <= 1-EPS]
        ##### Compute decision values #####
        dv = np.dot(alpha, kmat)
        risks = - dv
        ##### Update eta #####
        eta_new = np.zeros(NUM)
        ind_sorted = np.argsort(risks)[::-1]
        eta_new[ind_sorted[range(int(np.ceil(NUM*mu)))]] = 1.
        eta_new[ind_sorted[int(np.ceil(NUM*mu)-1)]] -= np.ceil(NUM*mu) - NUM*mu
        eta_new = 1 - eta_new
        if all(eta == eta_new):
            print 'CONVERGED: TOTAL_ITR =', i+1
            return c, eta, kmat
        else: eta = eta_new
    print 'OVER MAXIMUM ITERATION'
    return c, eta, kmat

def rknsvc_mip(x, y, nu, mu):
    print '----- Robust nu-SVM MIP -----'
    num, dim = x.shape
    bigM = 1e3
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Names of variables #####
    names_w = ['w%s' % i for i in range(dim)]
    names_u = ['u%s' % i for i in range(num)]
    names_xi = ['xi%s' % i for i in range(num)]
    ##### Set variables #####
    c.variables.add(names=['rho'], obj=[-(nu-mu)], lb=[-cplex.infinity])
    c.variables.add(names=names_w, lb=[-cplex.infinity]*dim)
    c.variables.add(names=['b'], lb=[-cplex.infinity])
    c.variables.add(names=names_xi, obj=[1./num]*num)
    c.variables.add(names=names_u, types='B'*num)
    ##### Set quadratic objective #####
    c.objective.set_quadratic_coefficients(zip(names_w, names_w, [1]*dim))
    ##### Set linear constraints #####
    ## e^T u <= mu m
    c.linear_constraints.add(lin_expr=[[names_u, [1]*num]], rhs=[mu*num], senses = 'L')
    for i in xrange(num):
        linexpr = [[range(dim+2) + [dim+2+i] + [dim+num+2+i], [-1] + list(x[i]*y[i]) + [y[i]] + [1] + [bigM]]]
        c.linear_constraints.add(lin_expr = linexpr, senses='G')
    ## c.write('test.lp')
    c.solve()
    print 'Status:', c.solution.get_status_string()
    print 'Objective Value:', c.solution.get_objective_value()
    print '\n'
    return c

if __name__ == '__main__':
    ## Synthetic data set including outliers
    ## np.random.seed(1)
    ## num_p, num_n, num_o = 200, 180, 15
    ## mu_p, mu_n, mu_o = [1,1], [-1,-1], [7,7]
    ## cov_p = [[2,0], [0,2]]
    ## cov_n = [[2,0], [0,2]]
    ## cov_o = [[5,0], [0,5]]
    ## x_p = np.random.multivariate_normal(mu_p, cov_p, num_p)
    ## x_n = np.random.multivariate_normal(mu_n, cov_n, num_n)
    ## x_o = np.random.multivariate_normal(mu_o, cov_o, num_o)
    ## x = np.vstack([x_p, x_n, x_o])
    ## y = np.array([1.]*num_p + [-1.]*(num_n+num_o))
    ## num, dim = x.shape
    ## print 'nu_min:', calc_nu_min(x, y)
    
    ## Read data set from csv
    ## dataset = np.loadtxt('Dataset/LIBSVM/splice/splice_scale.csv', delimiter=',')
    ## x = dataset[:, 1:]
    ## y = dataset[:, 0]
    ## num, dim = x.shape

    
    ## Names of variables
    ## names_w = ['w%s' % i for i in range(dim)]
    ## names_xi = ['xi%s' % i for i in range(num)]
    ## names_u = ['u%s' % i for i in range(num)]

    '''
    Numerical experiments for robust one-class SVM
    '''
    ## ###### Synthetic data set for one-class SVM #####
    ## np.random.seed(0)
    ## ## mean = [1,1]
    ## mean = [1, 1]
    ## cov = [[1,0],[0,1]]
    ## num = 200
    ## dim = 2
    ## x = np.random.multivariate_normal(mean ,cov, num)
    ## ##### Kernel function #####
    ## kernel = 'linear'
    ## ##### Hyper-parameters #####
    ## trial = 1
    ## ## nu_cand = np.arange(0.801, 0.9, 0.1)
    ## ## mu_cand = np.arange(0.4, 0.5, 0.1)
    ## nu_cand = np.array([0.8])
    ## mu_cand = np.array([0.2])
    ## kappa_cand = np.arange(10, 110, 10)
    ## dist = []
    ## for i in range(200):
    ##     for j in range(i+1, 200):
    ##         dist.append(np.linalg.norm(x[i] - x[j]))
    ## dist = np.array(dist)
    ## gamma = 1. / np.median(dist)**2
    ## ##### Robust one-class SVM #####
    ## df_result = pd.DataFrame(columns=['mu', 'nu', 'kappa', 'max_norm'])
    ## ##### Loop for mu #####
    ## for i in range(len(mu_cand)):
    ##     mu = mu_cand[i]
    ##     ##### Loop for nu #####
    ##     for j in range(len(nu_cand)):
    ##         nu = nu_cand[j]
    ##         ##### Loop for kappa #####
    ##         for k in range(len(kappa_cand)):
    ##             num_ol = kappa_cand[k]
    ##             ##### Initialize max_norm and max_rmse #####
    ##             max_norm = max_rmse = -1e10
    ##             for seed in range(trial):
    ##                 x_train = np.array(x)
    ##                 ## Generate noise
    ##                 np.random.seed(seed)
    ##                 ind_ol = np.random.choice(range(200), num_ol, replace=False)
    ##                 ## x_train[ind_ol] += np.random.multivariate_normal(np.zeros(2) ,1e2*np.eye(2), num_ol)
    ##                 x_train[ind_ol] = np.random.multivariate_normal(10*np.ones(2) ,1e1*np.eye(2), num_ol)
    ##                 if nu > mu:
    ##                     res_oc, eta, kmat = robust_ocsvm(x_train, nu, mu, kernel, gamma=gamma)
    ##                     alf = np.array(res_oc.solution.get_values())
    ##                     norm = np.sqrt(np.dot(alf, np.dot(kmat, alf)))# * (nu - mu) ###
    ##                     max_norm = np.max([max_norm, norm])
    ##                     ## ind_sv = np.where(alf != 0)[0]
    ##                     ## print ind_ol
    ##                     ## print alf
    ##                     ## plt.plot(x_train[:,0], x_train[:,1], 'x')
    ##                     ## plt.plot(x_train[ind_sv,0], x_train[ind_sv,1], 'o')
    ##                     ## plt.grid()
    ##                     ## plt.show()
    ##                     ##### Scikit-learn #####
    ##                     clf = svm.OneClassSVM(nu=nu, kernel="linear", gamma=0.1)
    ##                     clf.fit(x_train)
    ##                     print 'w:', -2*np.dot(x_train.T, alf), clf.coef_
    ##             df_result = df_result.append(pd.Series([mu, nu-mu, num_ol, max_norm], index=['mu', 'nu', 'kappa', 'max_norm']), ignore_index=True)

    ## #tmp = 5
    ## tmp2 = 'max_norm'
    ## plt.plot( df_result['kappa']/200., df_result['max_norm'], label='(nu, mu) = (0.5, 0.3)')
    ## plt.axvline(x=0.2, color='black')
    ## plt.grid()
    ## plt.xlabel('kappa/mu')
    ## plt.ylabel(tmp2)
    ## plt.legend()
    ## plt.show()

    '''
    Numerical experiments comparing the global solution by MIP
    and a local solution by DCA changing initial point
    '''
    ##### Hyper-parameters
    nu = np.array([0.41, 0.51, 0.61, 0.71, 0.81])
    mu = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    num_glob1 = np.zeros(len(nu))
    num_glob5 = np.zeros(len(nu))
    num_glob10 = np.zeros(len(nu))
    s_max = np.zeros(len(nu))
    ##### Loop for hyper-parameters
    for i in range(len(nu)):
        ##### Loop for synthetic dataset #####
        s = 0
        for j in range(100):
            while True:
                np.random.seed(s)
                dim = 10
                mean_p = np.ones(dim) * 3
                mean_n = np.ones(dim)
                cov = np.identity(dim) * 50
                num_p = num_n = 20
                x_p = np.random.multivariate_normal(mean_p, cov, num_p)
                x_n = np.random.multivariate_normal(mean_n, cov, num_n)
                x = np.vstack([x_p, x_n])
                y = np.array([1.]*num_p + [-1.]*num_n)
                num, dim = x.shape
                ##### Outliers #####
                ind_ol = np.random.choice(range(num_p), 4, replace=False)
                y[ind_ol] = y[ind_ol] * -1
                ##### Globally solve robust-nu-SVC #####
                res_mip = rknsvc_mip(x, y, nu[i], mu[i])
                obj_mip = res_mip.solution.get_objective_value()
                s += 1
                if obj_mip <= -1e-4:
                    break
                else:
                    print 'opt(MIP) =', obj_mip
                    print '(nu, mu) =', (nu[i], mu[i])
            ##### Loop for initial points #####
            obj_dc = np.zeros(10)
            for k in range(10):
                ##### Initial point #####
                np.random.seed(k)
                eta_init = np.ones(num)
                eta_init[np.random.choice(range(num), 4, replace=False)] = 0.
                ##### Solved by DCA #####
                res_dc, b_dc, rho_dc, eta_dc = robust_nusvm(x, y, nu[i], mu[i], kernel='linear', eta=eta_init)
                alf = np.array(res_dc.solution.get_values())
                w_dc = np.dot(x.T, alf*y)
                xi_dc = rho_dc - y * (np.dot(x, w_dc) + b_dc)
                xi_dc[xi_dc <= 0] = 0.
                obj_dc[k] = np.dot(eta_dc, xi_dc) / num + np.dot(w_dc, w_dc) / 2 - (nu[i]-mu[i])*rho_dc
            if obj_dc[0] / obj_mip >= 0.97:# or obj_dc[0] - obj_mip <= 1e-4:
                num_glob1[i] += 1
                num_glob5[i] += 1
                num_glob10[i] += 1
            elif min(obj_dc[:5]) / obj_mip >= 0.97:# or min(obj_dc[:5]) - obj_mip <= 1e-4:
                num_glob5[i] += 1
                num_glob10[i] += 1
            elif min(obj_dc) / obj_mip >= 0.97:# or min(obj_dc) - obj_mip <= 1e-4:
                num_glob10[i] += 1
        s_max[i] = s

    ## ##### Loop for synthetic data set #####
    ## test = []
    ## for s in range(10):
    ##     obj_dc = np.zeros([len(nu), trial])
    ##     obj_mip = np.zeros(len(nu))
    ##     dim = 2
    ##     mu1 = np.ones(dim) * 3
    ##     mu2 = np.ones(dim)
    ##     cov = np.identity(dim) * 1
    ##     num_p = num_n = 20
    ##     np.random.seed(0)
    ##     x_p = np.random.multivariate_normal(mu1,cov,num_p)
    ##     x_n = np.random.multivariate_normal(mu2,cov,num_n)
    ##     x = np.vstack([x_p, x_n])
    ##     y = np.array([1.]*num_p + [-1.]*num_n)
    ##     num, dim = x.shape
    ##     ##### Names of variables #####
    ##     names_w = ['w%s' % i for i in range(dim)]
    ##     names_xi = ['xi%s' % i for i in range(num)]
    ##     names_u = ['u%s' % i for i in range(num)]
    ##     ##### Loop for hyper-parameters #####
    ##     for i in range(len(nu)):
    ##         ##### Globally solve robust-nu-SVC #####
    ##         res_mip = rknsvc_mip(x, y, nu[i], mu[i])
    ##         w_mip = np.array(res_mip.solution.get_values(names_w))
    ##         b_mip = res_mip.solution.get_values('b')
    ##         xi_mip = np.array(res_mip.solution.get_values(names_xi))
    ##         eta_mip = 1 - np.array(res_mip.solution.get_values(names_u))
    ##         rho_mip = res_mip.solution.get_values('rho')
    ##         obj_mip[i] = res_mip.solution.get_objective_value()
    ##         ##### Loop for initial eta #####
    ##         for j in range(trial):
    ##             ##### Initial point #####
    ##             np.random.seed(j)
    ##             eta_init = np.ones(num)
    ##             eta_init[np.random.choice(range(num), 4, replace=False)] = 0.
    ##             ##### Solved by DCA #####
    ##             res_dc, b_dc, rho_dc, eta_dc = robust_nusvm(x, y, nu[i], mu[i], kernel='linear', eta=eta_init)
    ##             alf = np.array(res_dc.solution.get_values())
    ##             w_dc = np.dot(x.T, alf*y)
    ##             ## y_i (w x_i + b) >= rho - xi
    ##             xi_dc = rho_dc - y * (np.dot(x, w_dc) + b_dc)
    ##             xi_dc[xi_dc <= 0] = 0.
    ##             obj_dc[i, j] = np.dot(eta_dc, xi_dc) / num + np.dot(w_dc, w_dc) / 2 - (nu[i]-mu[i])*rho_dc
    ##     num_glob = np.array([sum(obj_dc[i]/obj_mip[i] > 0.97) for i in range(5)])
    ##     test.append(num_glob)

    ##### Box plot #####
    ## plt.boxplot([obj_dc[0]/obj_mip[0],
    ##              obj_dc[1]/obj_mip[1],
    ##              obj_dc[2]/obj_mip[2],
    ##              obj_dc[3]/obj_mip[3],
    ##              obj_dc[4]/obj_mip[4]], whis=500000)
    ## plt.ylim([0.8, 1.01])
    ## plt.xlabel('(nu, mu)')
    ## plt.ylabel('obj(DCA) / obj(MIP)')
    ## plt.grid()
    ## plt.xticks(range(1,6), ['(0.4, 0.1)', '(0.5, 0.1)',
    ##                        '(0.6, 0.1)', '(0.7, 0.1)', '(0.8, 0.1)'])
    ## plt.show()

    ## num_glob = np.array([sum(obj_dc[i]/obj_mip[i] > 1 - 1e-2) for i in range(5)])
    ## plt.xticks(range(0,5), ['(0.4, 0.1)', '(0.5, 0.1)',
    ##                        '(0.6, 0.1)', '(0.7, 0.1)', '(0.8, 0.1)'])
    ## plt.plot(num_glob)
    ## test = np.array(test)
    ## plt.boxplot([test[:,0],
    ##              test[:,1],
    ##              test[:,2],
    ##              test[:,3],
    ##              test[:,4]], whis=500000)
    ## plt.plot(test[0], 'x')
    ## plt.plot(test[1], 'x')
    ## plt.plot(test[2], 'x')
    ## plt.plot(test[3], 'x')
    ## plt.plot(test[4], 'x')
    ## plt.plot([max(obj_dc[i]/obj_mip[i]) for i in range(5)], label='max')
    ## plt.plot([np.mean(obj_dc[i]/obj_mip[i]) for i in range(5)], label='mean')
    ## plt.plot([min(obj_dc[i]/obj_mip[i]) for i in range(5)], label='min')
    ## plt.ylabel('#{obj(DCA) = obj(MIP)}')
    ## plt.xlabel('(nu, mu)')
    ## plt.ylim([0, 105])
    ## plt.legend()
    ## plt.grid()
    ## plt.show()

    ## plt.plot(x_p[:,0], x_p[:,1], 'o')
    ## plt.plot(x_n[:,0], x_n[:,1], '^')
    ## plt.show()


    '''
    Numerical experiments comparing the global solution by MIP
    and a local solution by DCA
    '''
    ## ##### Hyper-parameters
    ## nu = np.array([0.41, 0.51, 0.61, 0.71, 0.81])
    ## mu = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    ## trial = 1
    ## ##### Synthetic data set #####
    ## ## obj_dc, obj_mip = [], []
    ## obj_dc = np.zeros([len(nu), trial])
    ## obj_mip = np.zeros([len(nu), trial])
    ## ##### Loop for seed #####
    ## for s in range(trial):
    ##     np.random.seed(s)
    ##     mu1 = [2,2]
    ##     mu2 = [1,1]
    ##     cov = [[1,0],[0,1]]
    ##     num_p = num_n = 20
    ##     x_p = np.random.multivariate_normal(mu1,cov,num_p)
    ##     x_n = np.random.multivariate_normal(mu2,cov,num_n)
    ##     x = np.vstack([x_p, x_n])
    ##     y = np.array([1.]*num_p + [-1.]*num_n)
    ##     num, dim = x.shape
    ##     ##### Names of variables #####
    ##     names_w = ['w%s' % i for i in range(dim)]
    ##     names_xi = ['xi%s' % i for i in range(num)]
    ##     names_u = ['u%s' % i for i in range(num)]
    ##     ##### Loop for hyper-parameters #####
    ##     for i in range(len(nu)):
    ##         ##### Globally solve robust-nu-SVC #####
    ##         res_mip = rknsvc_mip(x, y, nu[i], mu[i])
    ##         ## print res_mip.solution.get_objective_value()
    ##         w_mip = np.array(res_mip.solution.get_values(names_w))
    ##         b_mip = res_mip.solution.get_values('b')
    ##         xi_mip = np.array(res_mip.solution.get_values(names_xi))
    ##         eta_mip = 1 - np.array(res_mip.solution.get_values(names_u))
    ##         rho_mip = res_mip.solution.get_values('rho')
    ##         ## obj_mip.append(res_mip.solution.get_objective_value())
    ##         obj_mip[i, s] = res_mip.solution.get_objective_value()
    ##         ##### Solved by DCA #####
    ##         res_dc, b_dc, rho_dc, eta_dc = robust_nusvm(x, y, nu[i], mu[i], kernel='linear')
    ##         alf = np.array(res_dc.solution.get_values())
    ##         w_dc = np.dot(x.T, alf*y)
    ##         ## y_i (w x_i + b) >= rho - xi
    ##         xi_dc = rho_dc - y * (np.dot(x, w_dc) + b_dc)
    ##         xi_dc[xi_dc <= 0] = 0.
    ##         ## obj_dc.append(np.dot(eta_dc, xi_dc) / num + np.dot(w_dc, w_dc) / 2 - (nu[i]-mu[i])*rho_dc)
    ##         obj_dc[i, s] = np.dot(eta_dc, xi_dc) / num + np.dot(w_dc, w_dc) / 2 - (nu[i]-mu[i])*rho_dc
    ## ##### Box plot #####
    ## ind_nonzero = [np.where(i < -1e-3)[0] for i in obj_mip]
    ## plt.boxplot([obj_dc[0,ind_nonzero[0]]/obj_mip[0,ind_nonzero[0]],
    ##              obj_dc[1,ind_nonzero[1]]/obj_mip[1,ind_nonzero[1]],
    ##              obj_dc[2,ind_nonzero[2]]/obj_mip[2,ind_nonzero[2]],
    ##              obj_dc[3,ind_nonzero[3]]/obj_mip[3,ind_nonzero[3]],
    ##              obj_dc[4,ind_nonzero[4]]/obj_mip[4,ind_nonzero[4]]], whis=500)
    ## ## plt.boxplot([obj_dc[0]/obj_mip[0], obj_dc[1]/obj_mip[1],
    ## ##              obj_dc[2]/obj_mip[2], obj_dc[3]/obj_mip[3],
    ## ##              obj_dc[4]/obj_mip[4]])
    ## plt.ylim([0.8, 1.01])
    ## plt.xlabel('(nu, mu)')
    ## plt.ylabel('obj(DCA) / obj(MIP)')
    ## plt.grid()
    ## plt.xticks(range(1,6), ['(0.4, 0.1)', '(0.5, 0.1)',
    ##                        '(0.6, 0.1)', '(0.7, 0.1)', '(0.8, 0.1)'])
    ## plt.show()

    ## np.savetxt('obj_dc.csv', obj_dc, fmt="%.15f", delimiter=',')
    ## np.savetxt('obj_mip.csv', obj_mip, fmt="%.15f", delimiter=',')

    ##### Plot hyper plane #####
    ## ## memo w1 x1 + w2 x2 + b = 0
    ## yoko = np.array([-0., 3.])
    ## tate_dc = -yoko * w_dc[0] / w_dc[1] - b_dc / w_dc[1]
    ## tate_mip = -yoko * w_mip[0] / w_mip[1] - b_mip / w_mip[1]
    ## plt.plot(x_p[:,0], x_p[:,1], 'x')
    ## plt.plot(x_n[:,0], x_n[:,1], '+')
    ## plt.plot(yoko, tate_dc, label='DCA')
    ## plt.plot(yoko, tate_mip, label='MIP')
    ## plt.legend()
    ## plt.grid()
    ## plt.show()

    '''
    Numerical experiments comparing the speed of MIP and DCA
    '''
    ## nu = 0.71
    ## mu = 0.1
    ## trial = 10
    ## mu1 = [2,2]
    ## mu2 = [1,1]
    ## cov = [[1,0],[0,1]]
    ## time_dca = np.zeros([5,trial])
    ## time_mip = np.zeros([5,trial])
    ## num_cand = [10, 20, 30, 40, 50]
    ## ##### Loop for the number of samples #####
    ## for i in range(len(num_cand)):
    ##     num_p = num_n = num_cand[i]/2
    ##     np.random.seed(0)
    ##     x_p = np.random.multivariate_normal(mu1,cov,num_p)
    ##     x_n = np.random.multivariate_normal(mu2,cov,num_n)
    ##     x = np.vstack([x_p, x_n])
    ##     y = np.array([1.]*num_p + [-1.]*num_n)
    ##     num, dim = x.shape
    ##     for j in range(trial):
    ##         start_mip = time.time()
    ##         res_mip = rknsvc_mip(x, y, nu, mu)
    ##         end_mip = time.time()
    ##         start_dca = time.time()
    ##         res_dc, b_dc, rho_dc, eta_dc = robust_nusvm(x, y, nu, mu, kernel='linear')
    ##         end_dca = time.time()
    ##         time_dca[i,j] = end_dca - start_dca
    ##         time_mip[i,j] = end_mip - start_mip

    ## params = {#'backend': 'ps',
    ##     'axes.labelsize': 24,
    ##     #'text.fontsize': 18,
    ##     #'legend.fontsize': 28,
    ##     'legend.fontsize': 24,
    ##     'xtick.labelsize': 14,
    ##     'ytick.labelsize': 14,
    ##     #'text.usetex': False,
    ## }
    ## plt.rcParams.update(params)
    ## ## err = np.random.normal(size=100)
    ## ## plt.plot(num_cand, np.mean(time_mip, axis=1), label='MIP', lw=5)
    ## ## plt.plot(num_cand, np.mean(time_dca, axis=1), label='DCA', lw=5)
    ## plt.errorbar(num_cand, np.mean(time_mip, axis=1), yerr=np.std(time_mip, axis=1),
    ##              label='MIP', lw=5, elinewidth=3, capsize=5)
    ## plt.errorbar(num_cand, np.mean(time_dca, axis=1), yerr=np.std(time_dca, axis=1),
    ##              label='DCA', lw=5, elinewidth=3, capsize=5)
    ## plt.xlabel('num of samples')
    ## plt.ylabel('time (sec)')
    ## plt.yscale('log')
    ## plt.grid()
    ## plt.legend(shadow=False, prop={'size': 18}, loc='upper left')
    ## plt.show()
