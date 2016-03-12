# -*- coding: utf-8 -*-

"""
Enu-SVM using CPLEX
"""

import time
import numpy as np
import cplex


class EnuSVM:

    def __init__(self, nu=0.5, update_rule='projection', max_iter=1000, lp_method=1, gamma=0.5):
        self.nu = nu
        self.lp_method = lp_method
        self.update_rule = update_rule
        self.max_iter = max_iter
        self.gamma = gamma
        self.status = None

    def _solve_convex(self, x, y):
        """
        Description
        -----------
        Slove convex case of Enu-SVM

        Parameters
        ----------
        x: array of training samples
        y: array of training labels

        Returns
        -------
        CPLEX object
        """
        num, dim = x.shape
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        c = cplex.Cplex()
        c.set_results_stream(None)
        # Set variables
        c.variables.add(names=['rho'], lb=[-cplex.infinity], obj=[-self.nu*num])
        c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
        c.variables.add(names=['b'], lb=[- cplex.infinity])
        c.variables.add(names=xi_names, obj=[1.]*num)
        # Set quadratic constraint
        qexpr = [range(1, dim+1), range(1, dim+1), [1]*dim]
        c.quadratic_constraints.add(quad_expr=qexpr, rhs=1, sense='L', name='norm')
        # Set linear constraints: w * y_i * x_i + b * y_i + xi_i - rho >= 0
        for i in xrange(num):
            linexpr = [[w_names+['b']+['xi%s' % i]+['rho'], list(x[i]*y[i])+[y[i], 1., -1]]]
            c.linear_constraints.add(names=['margin%s' % i], senses='G', lin_expr=linexpr)
        # Solve QCLP
        c.solve()
        return c

    def _solve_nonconvex(self, x, y, initial_weight):
        """
        Description
        -----------
        Solve non-convex case of Enu-SVM

        Parameters
        ----------
        x: array of training samples
        y: array of training labels
        initial_weight: array of initial value used on non-convex case

        Returns
        -------
        CPLEX object
        """
        num, dim = x.shape
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        # Set initial point
        w_tilde = np.array(initial_weight)
        # Cplex object
        c = cplex.Cplex()
        c.set_results_stream(None)
        # Set variables
        c.variables.add(names=['rho'], lb=[-cplex.infinity], obj=[-self.nu*num])
        c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
        c.variables.add(names=['b'], lb=[-cplex.infinity])
        c.variables.add(names=xi_names, obj=[1.]*num)
        # Set linear constraints: w * y_i * x_i + b * y_i + xi_i - rho >= 0
        c.parameters.lpmethod.set(self.lp_method)
        for i in xrange(num):
            c.linear_constraints.add(
                names=['margin%s' % i], senses='G',
                lin_expr=[[w_names+['b']+['xi'+'%s' % i]+['rho'], list(x[i]*y[i]) + [y[i], 1., -1]]]
            )
        # w_tilde * w = 1
        c.linear_constraints.add(names=['norm'], lin_expr=[[w_names, list(w_tilde)]], senses='E', rhs=[1.])
        # Iteration
        self.total_itr = 0
        for i in xrange(self.max_iter):
            self.total_itr += 1
            c.solve()
            self.weight = np.array(c.solution.get_values(w_names))
            # Termination
            if np.linalg.norm(self.weight - w_tilde) < 1e-5:
                self.status = 'converge'
                return c
            # Update norm constraint
            if self.update_rule == 'projection':
                w_tilde = self.weight / np.linalg.norm(self.weight)
            elif self.update_rule == 'lin_comb':
                w_tilde = self.gamma * w_tilde + (1-self.gamma) * self.weight
            c.linear_constraints.delete('norm')
            c.linear_constraints.add(names=['norm'], lin_expr=[[w_names, list(w_tilde)]], senses='E', rhs=[1.])

    # Training Enu-SVM
    def fit(self, x, y, initial_weight=None):
        """
        Description
        -----------
        Training Enu-SVM

        Parameters
        ----------
        x: array of training samples
        y: array of training labels
        initial_weight: array of initial value used on non-convex case

        Returns
        -------
        None
        """
        start = time.time()
        self.status = None
        num, dim = x.shape
        w_names = ['w%s' % i for i in range(dim)]
        xi_names = ['xi%s' % i for i in range(num)]
        result = self._solve_convex(x, y)
        if -1e-5 < result.solution.get_objective_value() < 1e-5:
            result = self._solve_nonconvex(x, y, initial_weight)
            self.convexity = False
        else:
            self.convexity = True
        end = time.time()
        self.comp_time = end - start
        self.weight = np.array(result.solution.get_values(w_names))
        self.xi = np.array(result.solution.get_values(xi_names))
        self.bias = result.solution.get_values('b')
        self.rho = result.solution.get_values('rho')

    def score(self, x, y):
        """
        Description
        -----------
        Compute accuracy using the model

        Parameters
        ----------
        x: array of test samples
        y: array of test labels

        Returns
        -------
        Accuracy in [0, 1]
        """
        num, dim = x.shape
        dv = np.dot(x, self.weight) + self.bias
        return sum(dv * y > 0) / float(num)

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
    # Load data set
    dataset = np.loadtxt('data/LIBSVM/liver-disorders/liver-disorders_scale.csv', delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape
    # Training
    np.random.seed(0)
    initial_weight = np.random.normal(size=dim)
    model = EnuSVM(nu=0.395, update_rule='lin_comb')
    model.fit(x, y, initial_weight)
