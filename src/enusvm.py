# -*- coding: utf-8 -*-

'''
todo: range -> xrange
'''

import sys
# for Ubuntu 64bit
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cplex
import time
from sklearn.metrics import pairwise_kernels


class EnuSVM:

    def __init__(self):
        self.nu = 0.5
        self.cplex_method = 0
        self.update_rule = 'projection'
        self.max_itr = 100

    def set_initial_weight(self, initial_weight):
        self.initial_weight = initial_weight

    def set_nu(self, nu):
        self.nu = nu

    ##### Convex case of Enu-SVM #####
    def solve_convex_primal(self, x, y):
        num, dim = x.shape
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        c = cplex.Cplex()
        c.set_results_stream(None)
        ##### Set variables #####
        c.variables.add(names=['rho'], lb=[-cplex.infinity], obj=[-self.nu*num])
        c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
        c.variables.add(names=['b'], lb=[- cplex.infinity])
        c.variables.add(names=xi_names, obj=[1.]*num)
        ##### Set quadratic constraint #####
        qexpr = [range(1,dim+1), range(1,dim+1), [1]*dim]
        c.quadratic_constraints.add(quad_expr=qexpr, rhs=1, sense='L', name='norm')
        ##### Set linear constraints #####
        # w * y_i * x_i + b * y_i + xi_i - rho >= 0
        for i in xrange(num):
            linexpr = [[w_names+['b']+['xi%s' % i]+['rho'],
                        list(x[i]*y[i]) + [y[i], 1., -1]]]
            c.linear_constraints.add(names=['margin%s' % i],
                                     senses='G', lin_expr=linexpr)
        # Solve QCLP
        c.solve()
        return c

# Non-convex case of Enu-SVM
    def solve_nonconvex(self, x, y):
        num, dim = x.shape
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        # Set initial point
        w_tilde = np.array(self.initial_weight)
        # Cplex object
        c = cplex.Cplex()
        c.set_results_stream(None)
        # Set variables
        c.variables.add(names=['rho'], lb=[-cplex.infinity], obj = [-self.nu*num])
        c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
        c.variables.add(names=['b'], lb=[-cplex.infinity])
        c.variables.add(names=xi_names, obj=[1.]*num)
        # Set linear constraints: w * y_i * x_i + b * y_i + xi_i - rho >= 0
        c.parameters.lpmethod.set(1)
        for i in xrange(num):
            c.linear_constraints.add(names=['margin%s' % i], senses='G',
                                     lin_expr = [[w_names+['b']+['xi'+'%s' % i]+['rho'], list(x[i]*y[i]) + [y[i], 1., -1]]])
        # w_tilde * w = 1
        c.linear_constraints.add(names=['norm'], lin_expr=[[w_names, list(w_tilde)]], senses='E', rhs=[1.])
        # Iteration
        self.total_itr = 0
        for i in xrange(self.max_itr):
            self.total_itr += 1
            c.solve()
            self.weight = np.array(c.solution.get_values(w_names))
            # Termination
            if np.linalg.norm(self.weight - w_tilde) < 1e-5:
                return c
            # Update norm constraint
            if self.update_rule == 'projection':
                w_tilde = self.weight / np.linalg.norm(self.weight)
            elif update_rule == 'lin_comb':
                w_tilde = self.gamma * w_tilde + (1-self.gamma) * self.weight
            else:
                'ERROR: Input valid update rule'
            c.linear_constraints.delete('norm')
            c.linear_constraints.add(names=['norm'],
                                     lin_expr=[[w_names, list(w_tilde)]],
                                     senses = 'E', rhs = [1.])

    # Training Enu-SVM
    def solve_enusvm(self, x, y):
        start = time.time()
        num, dim = x.shape
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        result = self.solve_convex_primal(x, y)
        if -1e-5 < result.solution.get_objective_value() < 1e-5:
            result = self.solve_nonconvex(x, y)
            self.convexity = False
        else:
            self.convexity = True
        end = time.time()
        self.comp_time = end - start
        self.weight = np.array(result.solution.get_values(w_names))
        self.xi = np.array(result.solution.get_values(xi_names))
        self.bias = result.solution.get_values('b')
        self.rho = result.solution.get_values('rho')

    ## ===== To be public method ======================================
    def show_result(self, d=5):
        print '===== RESULT ==============='
        print 'nu:\t\t\t', self.nu
        print 'weight:\t\t', np.round(self.weight, d)
        print 'bias:\t\t', self.bias
        print 'rho:\t\t', self.rho
        ## print 'obj val:\t', self.obj[-1]
        ## print 'itaration:\t', self.total_itr
        print 'convexity:\t', self.convexity
        print 'time:\t\t', self.comp_time
        ## print 'accuracy:\t'
        print '============================'

##### Training nu-SVM using primal #####
def nusvm_primal(x, y, nu):
    num, dim = x.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set names of variables #####
    w_names  = ['w%s' % i for i in range(dim)]
    xi_names = ['xi%s' % i for i in range(num)]
    ##### Set variables (w, b, xi, rho) #####
    c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
    c.variables.add(names=['b'], lb=[- cplex.infinity])
    c.variables.add(obj=[cost]*num, names=xi_names)
    c.variables.add(names=['rho'])
    ##### Set quadratic objective #####
    c.objective.set_quadratic_coefficients(zip(range(dim), range(dim), [1]*dim))
    ##### Set linear constraint w*y_i*x_i + b*y_i + xi_i - rho >= 0 for all i #####
    linexpr = [[range(dim+1)+[dim+1+i], list(x[i]*y[i])+[y[i]]+[1.]] for i in range(num)]
    c.linear_constraints.add(lin_expr=linexpr, senses='G'*num, rhs=[1]*num)
    ##### Solve QP #####
    c.parameters.qpmethod.set(0)
    c.solve()
    return c

if __name__ == '__main__': 
    ## Load data set
    dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    y = dataset[:,0]
    x = dataset[:,1:]
    num, dim = x.shape
    ## Training
    svm = EnuSVM()
    svm.set_nu(0.15)
    np.random.seed(0)
    svm.set_initial_weight(np.random.normal(size=dim))
    svm.solve_enusvm(x, y)
    svm.show_result()