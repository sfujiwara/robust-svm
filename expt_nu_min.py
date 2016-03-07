# -*- coding: utf-8 -*-

"""
Compute nu_min for data sets used in the experiment.
"""

import numpy as np
from fsvm import ersvmutil

filename = 'data/LIBSVM/liver-disorders/liver-disorders_scale.csv'
filename = 'data/LIBSVM/diabetes/diabetes_scale.csv'
filename = 'data/LIBSVM/adult/a1a.csv'
# filename = 'data/LIBSVM/cod-rna/cod-rna.csv'
# filename = 'datasets/LIBSVM/svmguide1/svmguide1.csv'
dataset = np.loadtxt(filename, delimiter=',')
x = dataset[:, 1:]
y = dataset[:, 0]
y[y == 0] = -1

ersvmutil.standard_scale(x)

nu_min = ersvmutil.calc_nu_min(x, y)
nu_max = ersvmutil.calc_nu_max(y)

print '(nu_min, nu_max):', np.round((nu_min, nu_max), 5)
