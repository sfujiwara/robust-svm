# -*- coding: utf-8 -*-
# Module for standard SVMs

# Set path to CPLEX module
import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # for Ubuntu 64bit
sys.path.append('C:\Program Files (x86)\IBM\ILOG\CPLEX_Studio125\cplex\python\\x86_win32') # for Windows 32bit

# Import libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cplex
import time
from sklearn.metrics import pairwise_kernels

class KernelDualSVM:
    def __init__(self):
        self.eps = 1e-5
        self.opt_method = 0
        self.kernel = 'linear'
        self.coef0 = 0
        self.cost = 1.
        self.degree = 2
        self.gamma = 1

    ##### Setters #####
    def set_cost(self, cost):
        self.cost = cost
        
    def set_kernel(self, kernel):
        self.kernel = kernel

    ## Used to find support vectors and margin vectors
    def set_epsilon(self, eps):
        self.eps = eps
        
    def set_optimization_method(self, opt_method):
        self.opt_method = opt_method

    ##### Training C-SVM using dual #####
    def solve_svm(self, x, y):
        num, dim = x.shape
        c = cplex.Cplex()
        c.set_results_stream(None)
        c.parameters.qpmethod.set(self.opt_method)
        ##### Compute kernel gram matrix #####
        if self.kernel == 'linear':
            kmat = pairwise_kernels(x, metric='linear')
        elif self.kernel == 'rbf':
            kmat = pairwise_kernels(x, x, metric='rbf', gamma=gamma)
        elif self.kernel == 'polynomial':
            kmat = pairwise_kernels(x, metric='polynomial', coef0=coef0, degree=degree)
        else: print 'Undefined Kernel!!'
        qmat = (kmat.T * y).T * y + 1e-8*np.eye(num)
        ##### Set variables #####
        c.variables.add(obj=[-1]*num, ub=[self.cost]*num)
        ##### Set linear constraint #####
        c.linear_constraints.add(lin_expr=[[range(num), list(y)]], senses='E')
        ##### Set quadratic objective #####
        c.objective.set_quadratic([[range(num), list(qmat[i])] for i in range(num)])
        ##### Solve QP #####
        c.solve()
        self.alpha = np.array(c.solution.get_values())
        ##### Compute bias #####
        self.ind_sv = np.array([j for j in xrange(num) if self.eps <= self.alpha[j]/self.cost])
        self.ind_mv = np.array([j for j in self.ind_sv if self.alpha[j]/self.cost <= 1-self.eps])
        self.b = (1 - np.dot(qmat[self.ind_mv][:,self.ind_sv], self.alpha[self.ind_sv])) * y[self.ind_mv]


##### Training C-SVM using primal #####
def csvm_primal(x, y, cost, qpmethod=0):
    num, dim = x.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set names of variables #####
    w_names  = ['w%s' % i for i in range(dim)]
    xi_names = ['xi%s' % i for i in range(num)]
    ##### Set variables (w, b, xi) #####
    c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
    c.variables.add(names=['b'], lb=[- cplex.infinity])
    c.variables.add(obj=[cost]*num, names=xi_names)
    ##### Set quadratic objective #####
    c.objective.set_quadratic_coefficients(zip(range(dim), range(dim), [1]*dim))
    ##### Set linear constraint w*y_i*x_i + b*y_i + xi_i >= 1 for all i #####
    linexpr = [[range(dim+1)+[dim+1+i], list(x[i]*y[i])+[y[i]]+[1.]] for i in range(num)]
    c.linear_constraints.add(lin_expr=linexpr, senses='G'*num, rhs=[1]*num)
    ##### Solve QP #####
    c.parameters.qpmethod.set(qpmethod)
    c.solve()
    return c

##### Convex case of Enu-SVM #####
def solve_convex_primal(xmat, y, nu):
    m, n = xmat.shape
    w_names = ['w'+'%s' % i for i in range(n)]
    xi_names = ['xi'+'%s' % i for i in range(m)]
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set variables #####
    c.variables.add(names = ['rho'], lb = [- cplex.infinity], obj = [- nu*m])
    c.variables.add(names = w_names, lb = [- cplex.infinity] * n)
    c.variables.add(names = ['b'], lb = [- cplex.infinity])
    c.variables.add(names = xi_names, obj = [1.] * m)
    ##### Set quadratic constraint #####
    qexpr = [range(1,n+1), range(1,n+1), [1]*n]
    c.quadratic_constraints.add(quad_expr=qexpr, rhs=1, sense='L', name='norm')
    ##### Set linear constraints #####
    # w * y_i * x_i + b * y_i + xi_i - rho >= 0
    for i in xrange(m):
        c.linear_constraints.add(names = ['margin'+'%s' % i], senses = 'G',
                                 lin_expr = [[w_names + ['b'] + ['xi'+'%s' % i] + ['rho'], list(xmat[i]*y[i]) + [y[i], 1., -1]]])
    # Solve QCLP
    c.solve()
    return c

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

# Non-convex case of Enu-SVM
def solve_nonconvex(xmat, y, nu, w_init, update_rule):
    gamma = 0.9
    m, n = xmat.shape
    w_names = ['w'+'%s' % i for i in range(n)]
    xi_names = ['xi'+'%s' % i for i in range(m)]
    # Set initial point
    w_tilde = w_init
    # Cplex object
    c = cplex.Cplex()
    c.set_results_stream(None)
    # Set variables
    c.variables.add(names = ['rho'], lb = [- cplex.infinity], obj = [- nu*m])
    c.variables.add(names = w_names, lb = [- cplex.infinity] * n)
    c.variables.add(names = ['b'], lb = [- cplex.infinity])
    c.variables.add(names = xi_names, obj = [1.] * m)
    # Set linear constraints: w * y_i * x_i + b * y_i + xi_i - rho >= 0
    c.parameters.lpmethod.set(1)
    for i in xrange(m):
        c.linear_constraints.add(names = ['margin%s' % i], senses = 'G',
                                 lin_expr = [[w_names + ['b'] + ['xi'+'%s' % i] + ['rho'], list(xmat[i]*y[i]) + [y[i], 1., -1]]])
    # w_tilde * w = 1
    c.linear_constraints.add(names=['norm'], lin_expr=[[w_names, list(w_tilde)]], senses='E', rhs=[1.])
    # Iteration
    for i in xrange(1000):
        c.solve()
        w = np.array(c.solution.get_values(w_names))
        # Termination
        if np.linalg.norm(w-w_tilde) < 1e-5: return c
        # Update norm constraint
        if update_rule == 'projection':
            w_tilde = w / np.linalg.norm(w)
        elif update_rule == 'lin_comb': w_tilde = gamma*w_tilde + (1-gamma)*w
        else: 'ERROR: Input valid update rule'
        c.linear_constraints.delete('norm')
        c.linear_constraints.add(names = ['norm'], lin_expr = [[w_names, list(w_tilde)]], senses = 'E', rhs = [1.])
    return c

# Training Enu-SVM
def enusvm(x, y, nu, w_init, update_rule='projection'):
    m, n = x.shape
    result_convex = solve_convex_primal(x, y, nu)
    if -1e-5 < result_convex.solution.get_objective_value() < 1e-5:
        print 'Solve Non-Convex Case'
        result_nonconvex = solve_nonconvex(x, y, nu, w_init, update_rule)
        return result_nonconvex
    else:
        print 'Solve Convex Case'
        return result_convex

if __name__ == '__main__': 
    ## Load data set
    dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    y = dataset[:,0]
    x = dataset[:,1:]
    num, dim = x.shape
    ## Training
    svm = KernelDualSVM()
    svm.solve_svm(x, y)
