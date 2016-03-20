# -*- coding: utf-8 -*-

import time
import numpy as np

import enusvm
import svmutil


class HeuristicLinearERSVM:

    def __init__(self, nu=0.5, gamma=0.1, max_iter=100, heuristic_termination=True):
        self.max_iter = max_iter
        self.nu = nu
        self.gamma = gamma
        self.heuristic_termination = heuristic_termination
        self.is_convex = None
        self.total_iter = None
        self.comp_time = None

    def fit(self, x, y, initial_weight):
        start = time.time()
        self.weight = np.array(initial_weight)
        num, dim = x.shape
        self.total_iter = 0
        self.is_convex = True
        self.bias = 0
        self.ind_active = np.arange(num)
        model_enusvm = enusvm.EnuSVM()
        for i in range(self.max_iter):
            # Update nu
            nu_i = (self.nu * (1-self.gamma)**i * num) / len(self.ind_active)
            # Check bounded or not
            nu_max = svmutil.calc_nu_max(y[self.ind_active])
            if nu_i > nu_max:
                self.stp = 'over nu_max'
                break
            # Solve subproblem if bounded
            self.total_iter += 1
            model_enusvm.nu = nu_i
            initial_weight = np.array(self.weight)
            model_enusvm.fit(x[self.ind_active], y[self.ind_active], initial_weight)
            # Check convexity
            if not model_enusvm.convexity:
                self.is_convex = False
            w_new = model_enusvm.weight
            self.bias = model_enusvm.bias
            # Heuristic termination (1e-4 or 1e-5 is better)
            if self.heuristic_termination and np.abs(1 - np.dot(self.weight, w_new)) < 1e-4:
                self.stp = 'Heuristic Termination'
                break
            self.weight = w_new
            # Update loss and active set
            loss = - (np.dot(x, self.weight) + self.bias) * y
            card_active = np.ceil(num * (1 - self.nu + self.nu*(1-self.gamma)**(i+1)))
            new_active_set = np.argsort(loss)[range(np.int(card_active))]
            ind_active_new = np.argsort(loss)[range(np.int(card_active))]
            # Terminate if active set does not change
            if set(self.ind_active) == set(ind_active_new):
                self.stp = 'VaR Minimization'
                break
            else: self.ind_active = ind_active_new
            self.initial_weight = self.weight
        end = time.time()
        self.comp_time = end - start

    def score(self, x_test, y_test):
        num, dim = x_test.shape
        dv = np.dot(x_test, self.weight) + self.bias
        return sum(dv * y_test > 0) / float(num)

    def f1_score(self, x_test, y_test):
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


if __name__ == '__main__':
    dataset = np.loadtxt('data/LIBSVM/liver-disorders/liver-disorders_scale.csv', delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape
    np.random.seed(0)
    initial_weight = np.random.normal(size=dim)
    initial_weight /= np.linalg.norm(initial_weight)
    svm = HeuristicLinearERSVM()
    svm.fit(x, y, initial_weight)
