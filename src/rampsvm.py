# -*- coding: utf-8 -*-

'''
todo
* Save support vectors for prediction using nonlinear kernel
'''

import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # Ubuntu
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cplex
import time
from sklearn.metrics import pairwise_kernels

class RampSVM():

    ## ===== Constructor =========================================== ##
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
    ## ============================================================= ##


    ## ===== Setters =============================================== ##
    def set_cost(self, cost):
        self.cost = cost

    def set_s(self, s):
        self.s = s

    def set_epsilon(self, eps):
        self.eps = eps
    ## ============================================================= ##


    ## ===== Solve ================================================= ##
    def solve_rampsvm(self, x, y):
        self.total_itr = 0
        start = time.time()
        num, dim = x.shape
        ##### Initial point #####
        self.beta = np.zeros(num)
        ##### Compute kernel gram matrix #####
        if self.kernel == 'linear':
            kmat = pairwise_kernels(x, metric='linear')
        elif self.kernel == 'rbf':
            kmat = pairwise_kernels(x, metric='rbf', gamma=self.gamma)
        elif self.kernel == 'polynomial':
            kmat = pairwise_kernels(x, metric='polynomial',
                                    coef0=self.coef0,
                                    degree=self.degree)
        else:
            print 'Undefined Kernel!!'
        qmat = (kmat.T * y).T * y + 1e-7*np.eye(num)
        qmat = np.round(qmat, 10)
        ##### CPLEX object #####
        c = cplex.Cplex()
        c.set_results_stream(None)
        ##### Set variables #####
        c.variables.add(obj=[-1]*num)
        ##### Set quadratic objective #####
        print 'set qp obj'
        c.objective.set_quadratic([[range(num), list(qmat[i])] for i in range(num)])
        print 'done'
        ##### Set linear constraint #####
        c.linear_constraints.add(lin_expr=[[range(num), list(y)]],
                                 senses='E', rhs=[0])
        ##### Set QP optimization method #####
        c.parameters.qpmethod.set(self.cplex_method)
        for i in xrange(self.max_itr):
            ## print i
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
            self.decision_values = wx_seq + self.bias
            ##### Update beta #####
            beta_new = np.zeros(num)
            beta_new[np.where(self.decision_values*y < self.s)] = self.cost
            if all(self.beta == beta_new):
                break
            else:
                self.beta = beta_new
        if self.kernel == 'linear':
            self.weight = np.dot(x.T, self.alpha * y)
        
        self.accuracy = sum(self.decision_values * y > 0) / float(num)
        end = time.time()
        self.comp_time = end - start
    ## ============================================================= ##


    ## ===== Evaluation measures =================================== ##
    def calc_accuracy_linear(self, x, y):
        return sum((np.dot(x, self.weight) + self.bias) * y > 0) / float(len(y))

    def calc_f_linear(self, x_test, y_test):
        num, dim = x_test.shape
        dv = np.dot(x_test, self.weight) + self.bias
        ind_p = np.where(y_test > 0)[0]
        ind_n = np.where(y_test < 0)[0]
        tp = sum(dv[ind_p] > 0)
        tn = sum(dv[ind_n] < 0)
        recall = float(tp) / len(ind_p)
        if tp == 0:
            precision = 0.
        else:
            precision = float(tp) / (len(ind_n) - tn + tp)
        if recall + precision == 0:
            return 0.
        else:
            return 2*recall*precision / (recall+precision)
    ## ============================================================= ##


    def show_result(self, d=5):
        print '===== RESULT ==============='
        print '(cost, s):\t', (self.cost, self.s)
        print 'kernel:\t\t', self.kernel
        if self.kernel == 'linear':
            print 'weight:\t\t', np.round(self.weight, d)
        print 'bias:\t\t', np.round(self.bias, d)
        print 'itaration:\t', self.total_itr
        print 'accuracy:\t', self.accuracy
        print 'time:\t\t', self.comp_time
        print '============================'


if __name__ == '__main__':
    ## Read data set from csv
    filename = 'liver-disorders_scale.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    x = dataset[:, 1:]
    y = dataset[:, 0]
    num, dim = x.shape
    ## Set hyper-parameters
    rampsvm = RampSVM()
    rampsvm.set_cost(1e1)
    ## rampsvm.set_epsilon(1e-10)
    rampsvm.solve_rampsvm(x, y)
    rampsvm.show_result(3)
    kmat = pairwise_kernels(x, metric='linear')
