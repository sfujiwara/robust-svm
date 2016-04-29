# -*- coding: utf-8 -*-

import time
import numpy as np
import cplex
import svmutil


class LinearERSVM:

    def __init__(self, nu=0.5, mu=0.05, max_iter=30, tol=1e-10):
        self.nu = nu
        self.mu = mu
        self.max_iter = max_iter
        self.t = []
        self.tol = tol
        self.obj = []
        self.constant_t = -1

    # To be private method
    def initialize_result(self):
        self.total_iter = 0
        self.weight = self.initial_weight
        self.bias = self.initial_bias
        self.obj = []
        self.t = []

    def update_eta(self):
        m = len(self.risks)
        # Indices sorted by ascent order of risks
        ind_sorted = np.argsort(self.risks)[::-1]
        self.eta = np.zeros(m)
        self.eta[ind_sorted[range(int(np.ceil(m*self.mu)))]] = 1.
        self.eta[ind_sorted[int(np.ceil(m*self.mu)-1)]] -= np.ceil(m*self.mu) - m*self.mu
        self.eta = 1 - self.eta

    def score(self, x, y):
        num, dim = x.shape
        dv = np.dot(x, self.weight) + self.bias
        # return sum(dv * y > 0 + 1e-9) / float(num)
        return sum(dv * y > 0) / float(num)

    def f1_score(self, x, y):
        dv = np.dot(x, self.weight) + self.bias
        ind_p = np.where(y > 0)[0]
        ind_n = np.where(y < 0)[0]
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

    def fit(self, x, y, initial_weight, initial_bias=0.):
        time_start = time.time()
        self.initial_weight = initial_weight
        self.initial_bias = initial_bias
        self.initialize_result()
        num, dim = x.shape
        c = cplex.Cplex()
        c.set_results_stream(None)
        w_names = ['w%s' % i for i in xrange(dim)]
        xi_names = ['xi%s' % i for i in xrange(num)]
        # Initialize risk
        self.risks = - y * (np.dot(x, self.weight) + self.bias)
        # Initialize eta
        self.update_eta()
        eta_bef = self.eta
        # Initialize t
        obj_val = num * (svmutil.calc_cvar(self.risks, 1 - self.nu) * self.nu -
                         svmutil.calc_cvar(self.risks, 1 - self.mu) * self.mu)
        self.obj.append(obj_val)
        if self.constant_t < -0.5:
            self.t.append(max(0, self.obj[-1] / 0.99))
        else:
            self.t.append(self.constant_t)
        # Set variables and objective function
        c.variables.add(names=w_names, lb=[-cplex.infinity]*dim, ub=[cplex.infinity]*dim)
        c.variables.add(names=['b'], lb=[-cplex.infinity], ub=[cplex.infinity])
        c.variables.add(names=xi_names, obj=[1.]*num, lb=[0.]*num, ub=[cplex.infinity]*num)
        c.variables.add(names=['alpha'], obj=[self.nu*num], lb=[-cplex.infinity], ub=[cplex.infinity])
        # Set quadratic constraint
        c.quadratic_constraints.add(name='norm', quad_expr=[w_names, w_names, [1.]*dim], rhs=1., sense='L')
        # Set linear constraints w*y_i*x_i + b*y_i + xi_i - alf >= 0
        linexpr = [[w_names + ['b', 'xi%s' % i, 'alpha'], list(x[i]*y[i])+[y[i], 1., 1.]] for i in range(num)]
        names = ['margin%s' % i for i in range(num)]
        c.linear_constraints.add(names=names, senses='G'*num, lin_expr=linexpr)
        # Iteration
        for i in xrange(self.max_iter):
            self.total_iter += 1
            # Update objective function
            c.objective.set_linear('b', np.dot(1.-self.eta, y))
            c.objective.set_linear(zip(w_names, np.dot(y*(1.-self.eta), x)-2*self.t[-1]*self.weight))
            # Solve sub-problem
            c.solve()
            # print 'feasibility:', c.solution.is_primal_feasible()
            self.weight = np.array(c.solution.get_values(w_names))
            # xi = np.array(c.solution.get_values(xi_names))
            self.bias = c.solution.get_values('b')
            self.alpha = c.solution.get_values('alpha')
            # Update risk
            self.risks = - y * (np.dot(x, self.weight) + self.bias)
            # Update eta
            self.update_eta()
            # Objective Value
            obj_val = num * (svmutil.calc_cvar(self.risks, 1 - self.nu) * self.nu -
                             svmutil.calc_cvar(self.risks, 1 - self.mu) * self.mu)
            self.obj.append(obj_val)
            # Update t
            if self.constant_t < -0.5:
                self.t.append(max(1e-5 + self.obj[-1]/0.999, 0.))
            else:
                self.t.append(self.constant_t)
            # Termination
            diff = (self.obj[-2] - self.obj[-1]) / (abs(self.obj[-1]) + 1e-7)
            if diff < self.tol:
                break
            eta_bef = self.eta
        time_end = time.time()
        self.comp_time = time_end - time_start


if __name__ == '__main__':
    # Read a UCI dataset
    # dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    # y = dataset[:, 0]
    # x = dataset[:, 1:]
    # num, dim = x.shape
    # Load MNIST data
    from sklearn.datasets import fetch_mldata
    import profile
    np.random.seed(0)
    mnist = fetch_mldata('MNIST original', data_home='data/sklearn')
    mnist1 = mnist.data[mnist.target == 1]  # size = 7877
    mnist7 = mnist.data[mnist.target == 7]  # size = 7293
    x = np.vstack([mnist1, mnist7]).astype(float)
    y = np.array([1] * len(mnist1) + [-1] * len(mnist7))
    # Random sampling
    idx_rand = np.random.choice(len(y), size=100)
    x = x[idx_rand]
    y = y[idx_rand]
    num, dim = x.shape
    # Train ER-SVM
    svm = LinearERSVM(nu=0.65, mu=0.05)
    profile.run('svm.fit(x, y, initial_weight=np.ones(dim), initial_bias=0.)')
