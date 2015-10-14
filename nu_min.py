# -*- coding: utf-8 -*-

import numpy as np
from src import ersvmutil

filename = 'datasets/LIBSVM/liver-disorders/liver-disorders_scale.csv'
filename = 'datasets/LIBSVM/diabetes/diabetes_scale.csv'
filename = 'datasets/LIBSVM/adult/a1a.csv'
# filename = 'datasets/LIBSVM/cod-rna/cod-rna.csv'
filename = 'datasets/LIBSVM/svmguide1/svmguide1.csv'
dataset = np.loadtxt(filename, delimiter=',')
x = dataset[:, 1:]
y = dataset[:, 0]
y[y == 0] = -1

ersvmutil.standard_scale(x)

nu_min = ersvmutil.calc_nu_min(x, y)
nu_max = ersvmutil.calc_nu_max(y)

print '(nu_min, nu_max):', np.round((nu_min, nu_max), 5)
