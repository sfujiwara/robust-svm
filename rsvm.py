# -*- coding: utf-8 -*-

import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # Ubuntu
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cplex
import time
from sklearn.metrics import pairwise_kernels

##### Ramp Loss SVM #####
# kernel = {'linear', 'rbf', 'polynomial'}
def ramp_svm(x, y, cost, s, kernel, gamma=1., coef0=0., degree=2):
    ##### Constant values #####
    EPS = 1e-5
    MAX_ITR = 30
    m, n = x.shape
    NUM, DIM = x.shape
    ##### Initial point #####
    beta = np.zeros(m)
    ##### Compute kernel gram matrix #####
    if kernel == 'linear':
        kmat = pairwise_kernels(x, metric='linear')
    elif kernel == 'rbf':
        kmat = pairwise_kernels(x, x, metric='rbf', gamma=gamma)
    elif kernel == 'polynomial':
        kmat = pairwise_kernels(x, metric='polynomial', coef0=coef0, degree=degree)
    else:
        print 'Undefined Kernel!!'
    #kmat = kmat + 1e-8*np.eye(m)
    qmat = (kmat.T * y).T * y + 1e-7*np.eye(m)
    qmat = np.round(qmat, 10)
    ##### CPLEX object #####
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set variables #####
    c.variables.add(obj=[-1]*m)
    ##### Set quadratic objective #####
    ## qmat_sparse = [cplex.SparsePair(ind=range(m), val=list(qmat[i])) for i in range(m)]
    ## c.objective.set_quadratic(qmat_sparse)
    print 'Set quadratic objective'
    c.objective.set_quadratic([[range(m), list(qmat[i])] for i in range(m)])
    ##### Set linear constraint #####
    c.linear_constraints.add(lin_expr=[[range(m), list(y)]], senses='E', rhs=[0])
    ##### Set QP optimization method #####
    c.parameters.qpmethod.set(1)
    ## c.parameters.barrier.crossover.set(2)
    for i in xrange(MAX_ITR):
        ##### Update constraints #####
        c.variables.set_lower_bounds(zip(range(m), list(-beta)))
        c.variables.set_upper_bounds(zip(range(m), list(cost-beta)))
        ##### Solve subproblem #####
        ## print 'Solve subproblem'
        c.solve()
        alpha = np.array(c.solution.get_values())
        ##### Compute bias and decision values #####
        mv_ind = [j for j in xrange(m) if EPS <= (alpha[j]/cost) <= 1-EPS]
        wx_seq = np.dot(alpha*y, kmat)
        bias_seq = (y - wx_seq)[mv_ind]
        print bias_seq
        bias = np.mean(bias_seq)
        dv = wx_seq + bias
        ##### Update beta #####
        beta_new = np.zeros(m)
        beta_new[np.where(dv*y < s)] = cost
        if all(beta == beta_new):
            print 'CONVERGED: TOTAL_ITR =', i+1
            return c, beta
        else: beta = beta_new
    print 'OVER MAXIMUM ITERATION'
    return c, beta

if __name__ == '__main__':
    # Read data set from csv
    filename = 'datasets/LIBSVM/cod-rna/cod-rna.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    x = dataset[:, 1:]
    y = dataset[:, 0]
    num, dim = x.shape

    np.random.seed(1)
    ind_train = np.random.choice(len(y), 6000, replace=False)
    num, dim = x.shape
    x_train = x[ind_train]
    y_train = y[ind_train]
    # Train Ramp Loss SVM
    print 'Ramp SVM'
    t1 = time.time()
    res, beta = ramp_svm(x_train, y_train, cost=1e0, s=-1., kernel='linear', degree=2, coef0=1.)
    alpha = np.array(res.solution.get_values())
    t2 = time.time()
    print 'TIME:', t2 - t1
    print np.dot(alpha*y_train, x_train)
