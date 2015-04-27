import ersvmutil
## -*- coding: utf-8 -*-

import time
import sys
## Ubuntu
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux')
import numpy as np
import matplotlib.pyplot as plt
import cplex

class HeuristicLinearERSVM():
    def __init__(self):
        self.max_itr = 100
        self.stopping_rule = True
        self.nu = 0.7
        self.mu = 0.1
        self.gamma = 0.01

    def set_initial_point(initial_weight, initial_bias):
        self.weight = initial_weight
        self.bias = initial_bias

    def set_nu(nu):
        self.nu = nu

    def set_mu(mu):
        self.mu = mu

    def set_gamma(gamma):
        self.gamma = gamma

    def solve_heuristics(self, x, y):
        num, dim = x.shape
        self.total_itr = 0
        self.weight = np.zeros(dim)
        self.bias = 0
        self.ind_active = np.arange(num)
        for i in range(self.max_itr):
            ##### Update nu #####
            nu_i = (self.nu * (1-self.gamma)**i * num) / len(self.ind_active)
            x_active = x[self.ind_active]
            y_active = y[self.ind_active]
            ##### Check bounded or not
            nu_max = ersvmutil.calc_nu_max(y_active)
            if nu_i > nu_max:
                self.stp = 'over nu_max'
                break
            ##### Solve subproblem if bounded
            self.total_itr += 1
            result = enusvm.enusvm(xmat[active_set], y[active_set],
                                   nu_i, w_init)
            w_new = np.array(result.solution.get_values(range(1, n+1)))
            b = result.solution.get_values(n+1)
            ##### Heuristic termination (1e-4 or 1e-5 is better) #####
            if heuristics and np.abs(1 - np.dot(w, w_new)) < 1e-4:
                print 'Heuristic Termination:', 1-np.dot(w, w_new)
                break
            else: w = w_new
            ##### Update loss and active set #####
            loss = - (np.dot(xmat, w) + b) * y
            card_active = np.ceil(m * (1 - nu + nu*(1-gamma)**(i+1)))
            new_active_set = np.argsort(loss)[range(np.int(card_active))]
            ##### Terminate if active set does not change #####
            if set(active_set) == set(new_active_set):
                print 'VaR Minimization'
                break
            else: active_set = new_active_set
            w_init = w
        print 'ITR:', i + 1
        #return [result, w, b, active_set, nu_i]

if __name__ == '__main__':
    dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape
    svm = HeuristicLinearERSVM()
    svm.solve_heuristics(x, y)
