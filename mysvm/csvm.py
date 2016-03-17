# -*- coding: utf-8 -*-
# Module for standard SVMs

# Import libraries
import numpy as np
import cplex
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

    def set_cost(self, cost):
        self.cost = cost
        
    def set_kernel(self, kernel):
        self.kernel = kernel

    ## Used to find support vectors and margin vectors
    def set_epsilon(self, eps):
        self.eps = eps
        
    def set_optimization_method(self, opt_method):
        self.opt_method = opt_method

    # Training C-SVM using dual
    def solve_svm(self, x, y):
        num, dim = x.shape
        c = cplex.Cplex()
        c.set_results_stream(None)
        c.parameters.qpmethod.set(self.opt_method)
        # Compute kernel gram matrix
        if self.kernel == 'linear':
            kmat = pairwise_kernels(x, metric='linear')
        elif self.kernel == 'rbf':
            kmat = pairwise_kernels(x, x, metric='rbf', gamma=gamma)
        elif self.kernel == 'polynomial':
            kmat = pairwise_kernels(x, metric='polynomial', coef0=coef0, degree=degree)
        else: print 'Undefined Kernel!!'
        qmat = (kmat.T * y).T * y + 1e-8*np.eye(num)
        # Set variables
        c.variables.add(obj=[-1]*num, ub=[self.cost]*num)
        ##### Set linear constraint #####
        c.linear_constraints.add(lin_expr=[[range(num), list(y)]], senses='E')
        ##### Set quadratic objective #####
        c.objective.set_quadratic([[range(num), list(qmat[i])] for i in range(num)])
        # Solve QP
        c.solve()
        self.alpha = np.array(c.solution.get_values())
        # Compute bias
        self.ind_sv = np.array([j for j in xrange(num) if self.eps <= self.alpha[j]/self.cost])
        self.ind_mv = np.array([j for j in self.ind_sv if self.alpha[j]/self.cost <= 1-self.eps])
        self.b = (1 - np.dot(qmat[self.ind_mv][:,self.ind_sv], self.alpha[self.ind_sv])) * y[self.ind_mv]


# Training C-SVM using primal
def csvm_primal(x, y, cost, qpmethod=0):
    num, dim = x.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    # Set names of variables
    w_names  = ['w%s' % i for i in range(dim)]
    xi_names = ['xi%s' % i for i in range(num)]
    # Set variables (w, b, xi)
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
    c.variables.add(obj=[1./nu]*num, names=xi_names)
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
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape
    ## Training
    svm = KernelDualSVM()
    svm.solve_svm(x, y)
