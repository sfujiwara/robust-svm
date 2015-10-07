# -*- coding: utf-8 -*-

import time
from sklearn.metrics import pairwise_kernels
import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # Ubuntu
import cplex
import numpy as np
import matplotlib.pyplot as plt
import ersvmutil
import pandas as pd

# Non-linear classification using kernel
# Minimize a difference of CVaR by DCA (using non-linear kernel)
class KernelErSvm():

    def __init__(self, nu, mu, kernel, gamma=1., coef0=1, degree=2):
        self.nu     = nu
        self.mu     = mu
        self.kernel = kernel
        self.gamma  = gamma
        self.coef0  = coef0
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
        start = time.time()
        num, dim = x.shape
        c = cplex.Cplex()
        c.set_results_stream(None)
        a_names  = ['a%s'  % i for i in range(num)]
        xi_names = ['xi%s' % i for i in range(num)]
        kmat = self.kernel_matrix(x)
        kmat += np.eye(num) * 1e-10
        ## Initialize risk, eta, and t
        risks = - y * (np.dot(kmat, a) + b)
        eta = self.update_eta(risks)
        obj_val = [ersvmutil.calc_cvar(risks, 1-self.nu) * self.nu -
                   ersvmutil.calc_cvar(risks, 1-self.mu) * self.mu]
        t = [max(0, obj_val[-1] / 0.999)]
        # Set variables and objective function
        c.variables.add(names=a_names, lb=[-cplex.infinity]*num, ub=[cplex.infinity]*num)
        c.variables.add(names = ['b'], lb = [-cplex.infinity], ub = [cplex.infinity])
        c.variables.add(names=xi_names, obj=[1./num]*num, lb=[0.]*num, ub = [cplex.infinity]*num)
        c.variables.add(names=['alpha'], obj=[self.nu], lb=[-cplex.infinity], ub=[cplex.infinity])
        # Set quadratic constraints
        # ||w||^2 <= 1 where w = Sum{a_j*Phi(x_j)} <--> aKa <= 1
        ind1 = []
        for i in range(num):
                ind1 = ind1 + ['a%s' % i] * num
        ind2 = ['a%s' % i for i in range(num)] * num
        val  = list(np.reshape(kmat, num*num))
        c.quadratic_constraints.add(name='norm', quad_expr=[ind1, ind2, val], rhs=1., sense='L')
        ## y_i(wx_i + b) + xi_i + alpha >= 0 where w = Sum{a_j*Phi(x_j)}
        for i in xrange(num):
            names = a_names + ['b'] + ['xi%s' % i] + ['alpha']
            coeffs = [y[i] * kmat[i,j] for j in range(num)] + [y[i]] + [1.] + [1.]
            c.linear_constraints.add(names=['margin%s' % i], senses='G', lin_expr=[[names, coeffs]])
        ## Save the problem as text file
        ## c.write('test.lp')
        # Iteration
        for i in xrange(30):
            ## print '\nIteration:\t', i+1
            print '|w|^2 = aKa:\t', np.dot(np.dot(a, kmat), a)
            # Update objective function
            # - Sum_i{1 - eta_i^k} * y_i * b
            c.objective.set_linear('b', np.dot(1-eta, y) / num)
            # w_k * w <--> a_k^T K a, 
            coeffs1 = np.array([sum((1-eta[k]) * y[k] * kmat[j,k] for k in range(num)) for j in range(num)]) / num
            # w_k * w <--> a_k^T K a, 
            coeffs2 = - 2 * t[-1] * np.dot(a, kmat)
            c.objective.set_linear(zip(a_names, coeffs1+coeffs2))
            # Solve subproblem
            c.solve()
            a = np.array(c.solution.get_values(a_names))
            rho = - c.solution.get_values('alpha')
            # xi = np.array(c.solution.get_values(xi_names))
            bias = c.solution.get_values('b')
            # Calculate risk: r_i(w_k, b_k)
            risks = - np.array([y[j] * (sum(a[k] * kmat[j,k] for k in range(num)) + bias) for j in range(num)])
            # Update eta
            eta = self.update_eta(risks)
            # Update t
            # Termination
            obj_val_new = ersvmutil.calc_cvar(risks, 1 - self.nu) * self.nu - \
                          ersvmutil.calc_cvar(risks, 1 - self.mu) * self.mu
            if self.kernel == 'polynomial':
                t.append(max(0., obj_val[-1]/0.999 + 1e-7))
            else:
                t.append(max(0., obj_val[-1]/0.999))
            diff_obj = np.abs((obj_val[-1] - obj_val_new) / obj_val[-1])
            if diff_obj < 1e-5: break
            obj_val.append(obj_val_new)
            #print 'WEIGHT:\t\t', np.round(np.dot(dmat.T, a), 3)
        ## Set results
        self.a = a
        self.bias = bias
        self.rho = rho
        self.eta = eta
        self.decision_values = (np.dot(kmat, a) + bias)
        self.training_error = np.mean(self.decision_values * y < 0)
        self.support_vectors = x
        self.t = t
        self.obj_val = obj_val
        self.comp_time = time.time() - start
        self.itr = i + 1


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
    np.random.seed(0)
    ## Load data set
    dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    x = dataset[:, 1:]
    ersvmutil.standard_scale(x)
    y = dataset[:, 0]
    num, dim = x.shape
    ## num_tr = 104
    ## num_t = 241
    num_tr = 173
    num_t = 172
    nu_cand = np.arange(0.71, 0.1, -0.05)
    # nu_cand = np.array([0.1])
    trial = 30
    trial = 100
    # trial = 1
    ## Initial point
    a_init_lin = np.random.uniform(low=-1, high=1, size=num_tr)
    a_init_pol = np.random.uniform(low=-1, high=1, size=num_tr)

    df_pol = pd.DataFrame()
    df_lin = pd.DataFrame()
    err_pol = np.zeros(len(nu_cand))
    err_lin = np.zeros(len(nu_cand))

    for i in range(len(nu_cand)):
        for j in xrange(trial):
            # Split indices to training, validation, and test set
            ind_rand = np.random.permutation(range(num))
            ind_tr = ind_rand[:num_tr]
            ind_t = ind_rand[num_tr:]
            x_tr = np.array(x[ind_tr])
            y_tr = np.array(y[ind_tr])
            outliers = ersvmutil.runif_sphere(radius=10, dim=dim, size=6)
            x_tr[:6] = outliers
            ## Initial point
            a_init = np.random.uniform(low=-1, high=1, size=num_tr)
            kmat = pairwise_kernels(x_tr, metric='linear')
            #a_init_lin = a_init_lin / np.sqrt(np.dot(a_init_lin, np.dot(kmat, a_init_lin)))
            a_init_lin = a_init / np.sqrt(np.dot(a_init, np.dot(kmat, a_init)))
            kmat = pairwise_kernels(x_tr, metric='polynomial', coef0=1, degree=2)
            #a_init_pol = a_init_pol / np.sqrt(np.dot(a_init_pol, np.dot(kmat, a_init_pol)))
            a_init_pol = a_init / np.sqrt(np.dot(a_init, np.dot(kmat, a_init)))
            ## Train Kernel ER-SVM
            kersvm = KernelErSvm(nu=nu_cand[i], mu=0.03, kernel='polynomial')
            start = time.time()
            kersvm.solve(x=x_tr, y=y_tr, a=a_init_pol, b=0.)
            end = time.time()
            dv = kersvm.calc_decision_value(x[ind_t])
            err_pol[i] += sum(dv * y[ind_t] > 0)
            row_tmp = {
                'nu'            : nu_cand[i],
                'kernel'        : 'polynomial',
                'trial'         : j,
                'training_time' : kersvm.comp_time,
                'training_error': sum(kersvm.calc_decision_value(x_tr) * y_tr < 0) / float(num_tr),
                'test_error'    : sum(dv * y[ind_t] < 0) / float(num_t),
                'comp_time'     : end - start,
                'rho'           : kersvm.rho,
                'obj_val'       : kersvm.obj_val[-1],
            }
            df_pol = df_pol.append(pd.Series(row_tmp, name=pd.datetime.today()))
            print 'Result:', row_tmp
            a_init_pol = np.array(kersvm.a)

            ## Train Linear ER-SVM
            kersvm = KernelErSvm(nu=nu_cand[i], mu=0.03, kernel='linear')
            start = time.time()
            kersvm.solve(x=x_tr, y=y_tr, a=a_init_lin, b=0.)
            end = time.time()
            dv = kersvm.calc_decision_value(x[ind_t])
            err_lin[i] += sum(dv * y[ind_t] < 0)
            row_tmp = {
                'nu'            : nu_cand[i],
                'kernel'        : 'linear',
                'trial'         : j,
                'training_time' : kersvm.comp_time,
                'training_error': sum(kersvm.calc_decision_value(x_tr) * y_tr < 0) / float(num_tr),
                'test_error'    : sum(dv * y[ind_t] < 0) / float(num_t),
                'comp_time'     : end - start,
                'rho'           : kersvm.rho,
                'obj_val'       : kersvm.obj_val[-1],
           }
            df_lin = df_lin.append(pd.Series(row_tmp, name=pd.datetime.today()))
            a_init_lin = np.array(kersvm.a)

    plt.plot(nu_cand, err_lin/241., label='Linear')
    plt.plot(nu_cand, 1-err_pol/241., label='Polynomial')
    plt.grid()
    plt.legend()
    plt.ylabel('Test Error')
    plt.show()

    df_pol.to_csv('kernel_pol_rev.csv', index=False)
    df_lin.to_csv('kernel_lin_rev.csv', index=False)
