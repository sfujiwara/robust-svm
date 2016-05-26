# -*- coding: utf-8 -*-

"""
Compute nu_min for data sets used in the experiment.
"""

import numpy as np
from mysvm import svmutil

filename = 'data/libsvm/liver-disorders/liver-disorders_scale.csv'
filename = 'data/libsvm/diabetes/diabetes_scale.csv'
filename = 'data/libsvm/adult/a1a.csv'
# filename = 'data/libsvm/cod-rna/cod-rna.csv'
# filename = 'datasets/libsvm/svmguide1/svmguide1.csv'
dataset = np.loadtxt(filename, delimiter=',')
x = dataset[:, 1:]
y = dataset[:, 0]
y[y == 0] = -1

svmutil.standard_scale(x)

nu_min = svmutil.calc_nu_min(x, y)
nu_max = svmutil.calc_nu_max(y)

print '(nu_min, nu_max):', np.round((nu_min, nu_max), 5)
