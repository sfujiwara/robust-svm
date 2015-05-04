## -*- coding: utf-8 -*-

import sys
## Ubuntu
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux')
import numpy as np
import matplotlib.pyplot as plt
import cplex
import time
from src_old import ersvm

if __name__ == '__main__':
    ## Read a UCI dataset
    filename = 'datasets/LIBSVM/liver-disorders/liver-disorders_scale.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape

    ##### Names of variables #####
    w_names = ['w%s'  % i for i in range(dim)]
    xi_names = ['xi%s' % i for i in range(num)]

    ## Hyper-parameters
    nu = 0.8
    gamma = 0.01 #0.03/nu

    ## Initial weight
    np.random.seed(0)
    w_init = np.random.normal(size=dim)
    w_init = w_init / np.linalg.norm(w_init)

    ##### Heuristic algorithm #####
    res_h, w_h, b_h, active_set, nu_i = ersvm.heuristic_dr(x, y, nu, w_init, gamma=gamma, heuristics=True)
    risks_h = - (np.dot(x, w_h) + b_h) * y
    mu = 1 - len(active_set) / float(num)
    print w_h, b_h

    ##### Difference of CVaRs with linear kernel #####
    ## result, eta = diff_cvar(x, y, w_init, b_init, nu, mu)
    ## w_dc = np.array(result.solution.get_values(w_names))
    ## xi     = np.array(result.solution.get_values(xi_names))
    ## b_dc   = result.solution.get_values('b')
    ## alpha = result.solution.get_values('alpha')
    ## risks_dc = - y * (np.dot(x, w_dc) + b_dc)
