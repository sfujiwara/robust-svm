# -*- coding: utf-8 -*-

import time
from sklearn.metrics import pairwise_kernels
import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # Ubuntu
import cplex
import numpy as np
import matplotlib.pyplot as plt
import ersvmutil

# Non-linear classification using kernel
# Minimize a difference of CVaR by DCA (using non-linear kernel)
class KernelErSvm():

    def __init__(self, nu, mu, kernel, gamma=1., coef0=1, degree=2):
        self.nu = nu
        self.mu = mu
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def kernel_matrix(self, x):
        ##### Compute kernel gram matrix #####
        if self.kernel == 'linear':
            kmat = pairwise_kernels(x, metric='linear')
        elif self.kernel == 'rbf':
            kmat = pairwise_kernels(x, metric='rbf', gamma=self.gamma)
        elif self.kernel == 'polynomial':
            kmat = pairwise_kernels(x, metric='polynomial',
                                    coef0=self.coef0, degree=self.degree)
        else:
            print 'Undefined Kernel!!'
        return kmat


    def update_eta(self, risks):
        m = len(risks)
        ## Indices sorted by ascent order of risks
        ind_sorted = np.argsort(risks)[::-1]
        eta = np.zeros(m)
        eta[ind_sorted[range(int(np.ceil(m*self.mu)))]] = 1.
        eta[ind_sorted[int(np.ceil(m*self.mu)-1)]] -= np.ceil(m*self.mu) - m*self.mu
        eta = 1 - eta
        return eta
        

    def solve(self, x, y, a, b):
        num, dim = x.shape
        c = cplex.Cplex()
        c.set_results_stream(None)
        a_names  = ['a%s'  % i for i in range(num)]
        xi_names = ['xi%s' % i for i in range(num)]
        kmat = self.kernel_matrix(x)
        kmat += np.eye(num) * 1e-10

        # Initialize risk
        risks = - y * (np.dot(kmat, a) + b)
        # Initialize eta
        eta = self.update_eta(risks)
        ##### Initialize t #####
        obj_val = [num * (ersvmutil.calc_cvar(risks, 1-self.nu) * self.nu -
                          ersvmutil.calc_cvar(risks, 1-self.mu) * self.mu)]
        # Initialize t
        t = [max(0, obj_val[-1] / 0.99)]
        # Set variables and objective function
        c.variables.add(names=a_names, lb=[-cplex.infinity]*num, ub=[cplex.infinity]*num)
        c.variables.add(names = ['b'], lb = [-cplex.infinity], ub = [cplex.infinity])
        c.variables.add(names=xi_names, obj=[1./num]*num, lb=[0.]*num, ub = [cplex.infinity]*num)
        c.variables.add(names=['alpha'], obj=[self.nu], lb=[-cplex.infinity], ub=[cplex.infinity])
        # Set quadratic constraints
        # ||w||^2 <= 1 where w = Sum{a_j*Phi(x_j)} <--> aKa <= 1
        print 'Bottole neck?'
        ind1 = []
        for i in range(num):
                ind1 = ind1 + ['a%s' % i] * num
        print 'End bottle neck?'
        ind2 = ['a%s' % i for i in range(num)] * num
        val  = list(np.reshape(kmat, num*num))
        print '===== Set quadratic constraint ====='
        c.quadratic_constraints.add(name='norm', quad_expr=[ind1, ind2, val], rhs=1., sense='L')
        print '===================================='
        print '===== Set linear constraints ====='
        ## y_i(wx_i + b) + xi_i + alpha >= 0 where w = Sum{a_j*Phi(x_j)}
        for i in xrange(num):
            names = a_names + ['b'] + ['xi%s' % i] + ['alpha']
            coeffs = [y[i] * kmat[i,j] for j in range(num)] + [y[i]] + [1.] + [1.]
            c.linear_constraints.add(names=['margin%s' % i], senses='G', lin_expr=[[names, coeffs]])
        print '=================================='
        ## Save the problem as text file
        ## c.write('test.lp')
        # Iteration
        for i in xrange(20):
            print '\nIteration:\t', i+1
            #print '|w|^2 = aKa:\t', np.dot(np.dot(a, kmat), a)
            # Update objective function
            # - Sum_i{1 - eta_i^k} * y_i * b
            print 'Begin to update objective for b'
            print np.dot(1-eta, y) / num
            c.objective.set_linear('b', np.dot(1-eta, y) / num)
            print 'Finished to update objective'
            # w_k * w <--> a_k^T K a, 
            #coeffs1 = np.array([sum((1-eta[k]) * labels[k] * kernel(dmat[j], dmat[k]) for k in range(m)) for j in range(m)]) / m
            print 'Begin to update objective for w'
            coeffs1 = np.array([sum((1-eta[k]) * y[k] * kmat[j,k] for k in range(num)) for j in range(num)]) / num
            # w_k * w <--> a_k^T K a, 
            coeffs2 = - 2 * t[-1] * np.dot(a, kmat)
            c.objective.set_linear(zip(a_names, coeffs1+coeffs2))
            print 'Finished to update objective'
            # Solve subproblem
            print 'start'
            c.solve()
            print 'end'
            a = np.array(c.solution.get_values(a_names))
            # xi = np.array(c.solution.get_values(xi_names))
            bias = c.solution.get_values('b')
            # alpha = c.solution.get_values('alpha')
            # Calculate risk: r_i(w_k, b_k)
            risks = - np.array([y[j] * (sum(a[k] * kmat[j,k] for k in range(num)) + bias) for j in range(num)])
            # Update eta
            eta = self.update_eta(risks)
            # Update t
            # Termination
            obj_val_new = ersvmutil.calc_cvar(risks, 1-self.nu) * self.nu - ersvmutil.calc_cvar(risks, 1-self.mu) * self.mu
            t.append(max(0., obj_val[-1]/0.9))

            diff_obj = np.abs((obj_val[-1] - obj_val_new) / obj_val[-1])
            if diff_obj < 1e-5: break
            obj_val.append(obj_val_new)
            #print 'WEIGHT:\t\t', np.round(np.dot(dmat.T, a), 3)
        ## Set results
        self.a = a
        self.bias = bias
        self.eta = eta
        self.decision_values = (np.dot(kmat, a) + bias)
        self.training_error = np.mean(self.decision_values * y < 0)
        self.support_vectors = x


    def calc_decision_value(self, x):
        if self.kernel == 'linear':
            kmat = pairwise_kernels(self.support_vectors, x, metric=self.kernel)
        elif self.kernel == 'polynomial':
            kmat = pairwise_kernels(self.support_vectors, x,
                                    metric=self.kernel, coef0=self.coef0, degree=self.degree)
        else:
            print 'Undefined kernel!!'
        dv = np.dot(self.a, kmat) + self.bias
        return dv


if __name__ == '__main__':
    import pandas as pd
    np.random.seed(0)
    ## Load data set
    dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    x = dataset[:, 1:]
    ersvmutil.standard_scale(x)
    y = dataset[:, 0]
    num, dim = x.shape
    num_tr = 104
    num_t = 241
    nu_cand = np.arange(0.71, 0.1, -0.05)
    trial = 1
    ## Initial point
    a_init = np.random.uniform(low=-1, high=1, size=num_tr)

    df_dca = pd.DataFrame(columns=['trial', 'nu', 'mu', 'val-acc', 'val-f',
                                   'test-acc', 'test-f', 'VaR', 'tr-CVaR', 'comp_time'])
    err_pol = np.zeros(len(nu_cand))
    err_lin = np.zeros(len(nu_cand))
    for i in range(len(nu_cand)):
        for j in xrange(trial):
            # Split indices to training, validation, and test set
            ind_rand = np.random.permutation(range(num))
            ind_tr = ind_rand[:num_tr]
            ind_t = ind_rand[num_tr:]
            ## Train Kernel ER-SVM
            kersvm = KernelErSvm(nu=nu_cand[i], mu=0.03, kernel='polynomial')
            kersvm.solve(x=x[ind_tr], y=y[ind_tr], a=a_init, b=0.)
            dv = kersvm.calc_decision_value(x[ind_t])
            err_pol[i] += sum(dv * y[ind_t] > 0)
            print 'Training Error:', sum(kersvm.calc_decision_value(x[ind_tr]) * y[ind_tr] < 0) / float(num_tr)
            print 'Test Error:', sum(dv * y[ind_t] < 0) / float(num_t)
            ## Train Linear ER-SVM
            kersvm = KernelErSvm(nu=nu_cand[i], mu=0.03, kernel='linear')
            kersvm.solve(x=x[ind_tr], y=y[ind_tr], a=a_init, b=0.)
            dv = kersvm.calc_decision_value(x[ind_t])
            err_lin[i] += sum(dv * y[ind_t] < 0)

    plt.plot(nu_cand, err_lin/241., label='Linear')
    plt.plot(nu_cand, 1-err_pol/241., label='Polynomial')
    plt.grid()
    plt.legend()
    plt.ylabel('Test Error')
    plt.show()
