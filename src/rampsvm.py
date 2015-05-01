# -*- coding: utf-8 -*-

import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # Ubuntu
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cplex
import time
from sklearn.metrics import pairwise_kernels

class RampSVM():

    def __init__(self):
        self.eps = 1e-5
        self.max_itr = 30
        self.cost = 1e0
        self.s = -1.
        self.kernel = 'linear'
        self.gamma = 1.
        self.coef0 = 0
        self.degree = 2
        self.cplex_method = 1

    def set_cost(self, cost):
        self.cost = cost

    def set_epsilon(self, eps):
        self.eps = eps

    def solve_rampsvm(self, x, y):
        self.total_itr = 0
        num, dim = x.shape
        ##### Initial point #####
        self.beta = np.zeros(num)
        ##### Compute kernel gram matrix #####
        if self.kernel == 'linear':
            kmat = pairwise_kernels(x, metric='linear')
        elif self.kernel == 'rbf':
            kmat = pairwise_kernels(x, x, metric='rbf',
                                    gamma=self.gamma)
        elif self.kernel == 'polynomial':
            kmat = pairwise_kernels(x, metric='polynomial',
                                    coef0=self.coef0,
                                    degree=self.degree)
        else:
            print 'Undefined Kernel!!'
        #kmat = kmat + 1e-8*np.eye(m)
        qmat = (kmat.T * y).T * y + 1e-8*np.eye(num)
        qmat = np.round(qmat, 10)
        ##### CPLEX object #####
        c = cplex.Cplex()
        c.set_results_stream(None)
        ##### Set variables #####
        c.variables.add(obj=[-1]*num)
        ##### Set quadratic objective #####
        c.objective.set_quadratic([[range(num), list(qmat[i])] for i in range(num)])
        ##### Set linear constraint #####
        c.linear_constraints.add(lin_expr=[[range(num), list(y)]],
                                 senses='E', rhs=[0])
        ##### Set QP optimization method #####
        c.parameters.qpmethod.set(self.cplex_method)
        for i in xrange(self.max_itr):
            ##### Update constraints #####
            c.variables.set_lower_bounds(zip(range(num), list(-self.beta)))
            c.variables.set_upper_bounds(zip(range(num), list(self.cost-self.beta)))
            ##### Solve subproblem #####
            c.solve()
            self.total_itr += 1
            self.alpha = np.array(c.solution.get_values())
            ##### Compute bias and decision values #####
            ind_mv = [j for j in xrange(num) if self.eps <= (self.alpha[j]/self.cost) <= 1-self.eps]
            wx_seq = np.dot(self.alpha*y, kmat)
            bias_seq = (y - wx_seq)[ind_mv]
            ## print bias_seq
            self.bias = np.mean(bias_seq)
            dv = wx_seq + self.bias
            ##### Update beta #####
            beta_new = np.zeros(num)
            beta_new[np.where(dv*y < self.s)] = self.cost
            if all(self.beta == beta_new):
                break
            else:
                self.beta = beta_new
        self.weight = np.dot(x.T, self.alpha * y)

if __name__ == '__main__':
    ## Read data set from csv
    dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    x = dataset[:, 1:]
    y = dataset[:, 0]
    num, dim = x.shape
    ## Set hyper-parameters
    rampsvm = RampSVM()
    rampsvm.set_cost(1e1)
    ## rampsvm.set_epsilon(1e-10)
    rampsvm.solve_rampsvm(x, y)
