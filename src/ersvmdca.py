## -*- coding: utf-8 -*-

import time
import sys
## Ubuntu
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux')
import numpy as np
import matplotlib.pyplot as plt
import cplex

## import my modules
import ersvmutil

class LinearPrimalERSVM():
    ## ===== Constructor ==============================================
    def __init__(self):
        self.max_itr = 30
        self.cplex_method = 0
        self.t = []
        self.nu = 0.5
        self.mu = 0.1
        self.eps = 1e-10
        self.obj = []
        self.constant_t = -1
    ## ================================================================

    ## ===== Setters ==================================================
    def set_initial_point(self, initial_weight, initial_bias):
        self.initial_weight = initial_weight
        self.initial_bias = initial_bias

    def set_nu(self, nu):
        self.nu = nu

    def set_mu(self, mu):
        self.mu = mu

    def set_cplex_method(cplex_method):
        self.cplex_method = cplex_method

    def set_epsilon(self, eps):
        self.eps = eps

    def set_constant_t(self, constant_t):
        self.constant_t = constant_t
    ## ================================================================

    ## ===== To be private method =====================================
    ## Update self.eta using self.risks
    def initialize_result(self):
        self.total_itr = 0
        self.weight = self.initial_weight
        self.bias = self.initial_bias
        self.obj = []
        self.t = []

    def update_eta(self):
        m = len(self.risks)
        ## Indices sorted by ascent order of risks
        ind_sorted = np.argsort(self.risks)[::-1]
        self.eta = np.zeros(m)
        self.eta[ind_sorted[range(int(np.ceil(m*self.mu)))]] = 1.
        self.eta[ind_sorted[int(np.ceil(m*self.mu)-1)]] -= np.ceil(m*self.mu) - m*self.mu
        self.eta = 1 - self.eta
    ## ================================================================

    ## ===== To be public method ======================================
    def show_result(self, d=5):
        print '===== RESULT ==============='
        print '(nu, mu):\t', (self.nu, self.mu)
        print 'weight:\t\t', np.round(self.weight, d)
        print 'bias:\t\t', np.round(self.bias, d)
        print 'obj val:\t\t', self.obj[-1]
        print 'itaration:\t', self.total_itr
        print 'time:\t\t', self.comp_time
        print 'accuracy:\t', sum(self.risks < 0) / float(len(self.risks))
        print '============================'
        
    def solve_ersvm(self, x, y):
        time_start = time.time()
        self.initialize_result()
        num, dim = x.shape
        c = cplex.Cplex()
        c.set_results_stream(None)
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        ##### Initialize risk #####
        self.risks = - y * (np.dot(x, self.weight) + self.bias)
        ##### Initialize eta #####
        self.update_eta()
        eta_bef = self.eta
        ##### Initialize t #####
        obj_val = num * (ersvmutil.calc_cvar(self.risks, 1-self.nu) * self.nu -
                         ersvmutil.calc_cvar(self.risks, 1-self.mu) * self.mu)
        self.obj.append(obj_val)
        if self.constant_t < -0.5:
            self.t.append(max(0, self.obj[-1] / 0.99))
        else:
            self.t.append(self.constant_t)
        ##### Set variables and objective function #####
        c.variables.add(names=w_names, lb=[-cplex.infinity]*dim, ub=[cplex.infinity]*dim)
        c.variables.add(names=['b'], lb=[-cplex.infinity], ub=[cplex.infinity])
        c.variables.add(names=xi_names, obj=[1.]*num,
                        lb=[0.]*num, ub=[cplex.infinity]*num)
        c.variables.add(names=['alpha'], obj=[self.nu * num],
                        lb=[-cplex.infinity], ub=[cplex.infinity])
        ##### Set quadratic constraint #####
        c.quadratic_constraints.add(name='norm', quad_expr=[w_names, w_names, [1.]*dim], rhs=1., sense='L')
        ##### Set linear constraints w*y_i*x_i + b*y_i + xi_i - alf >= 0 #####
        linexpr = [[w_names + ['b', 'xi%s' % i, 'alpha'],
                    list(x[i] * y[i])+[y[i], 1., 1.]] for i in range(num)]
        names = ['margin%s' % i for i in range(num)]
        c.linear_constraints.add(names=names, senses='G'*num, lin_expr=linexpr)
        ##### Set QP optimization method #####
        c.parameters.qpmethod.set(self.cplex_method)
        ##### Iteration #####
        for i in xrange(self.max_itr):
            self.total_itr += 1
            ##### Update objective function #####
            c.objective.set_linear('b', np.dot(1 - self.eta, y))
            c.objective.set_linear(zip(w_names,
                                       np.dot(y*(1-self.eta), x) - 2*self.t[-1]*self.weight))        
            ##### Solve subproblem #####
            c.solve()
            self.weight = np.array(c.solution.get_values(w_names))
            ## xi = np.array(c.solution.get_values(xi_names))
            self.bias = c.solution.get_values('b')
            self.alpha = c.solution.get_values('alpha')
            ##### Update risk #####
            self.risks = - y * (np.dot(x, self.weight) + self.bias)
            ##### Update eta #####
            self.update_eta()
            ##### Objective Value #####
            obj_val = num * (ersvmutil.calc_cvar(self.risks, 1-self.nu) * self.nu -
                             ersvmutil.calc_cvar(self.risks, 1-self.mu) * self.mu)
            self.obj.append(obj_val)
            ##### Update t #####
            if self.constant_t < -0.5:
                self.t.append(max(1e-5 + self.obj[-1]/0.999, 0.))
            else:
                self.t.append(self.constant_t)
            ##### Termination #####
            if (self.obj[-2] - self.obj[-1] < self.eps): break
            eta_bef = self.eta
        time_end = time.time()
        self.comp_time = time_end - time_start

if __name__ == '__main__':
    ## Read a UCI dataset
    dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape
    svm = LinearPrimalERSVM()
    svm.set_initial_point(np.ones(dim), 0)
    svm.set_nu(0.65)
    svm.set_mu(0.1)
    svm.set_constant_t(1.5)
    svm.solve_ersvm(x, y)
    svm.show_result(5)
