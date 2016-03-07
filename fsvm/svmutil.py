## -*- coding: utf-8 -*-

import sys, time
## Ubuntu
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux')
import numpy as np
import matplotlib.pyplot as plt
import cplex
from sklearn.metrics import pairwise_kernels


def standard_scale(x):
    num, dim = x.shape
    for i in range(dim):
        x[:,i] -= np.mean(x[:,i])
        if np.std(x[:,i]) >= 1e-5:
            x[:,i] /= np.std(x[:,i])


def libsvm_scale(x):
    num, dim = x.shape
    for i in range(dim):
        width = max(x[:,i]) - min(x[:,i])
        x[:,i] /= (width / 2)
        x[:,i] -= max(x[:,i]) - 1


##### Compute nu_max #####
def calc_nu_max(y):
    return np.double(len(y)-np.abs(np.sum(y))) / len(y)


##### Compute nu_min for linear kernel #####
## nu_min = 0 の場合解が存在しないので, 対応させること
def calc_nu_min(xmat, y):
    m, n = xmat.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    c.variables.add(obj=[1]+[0]*m)
    c.linear_constraints.add(lin_expr=[[range(1,m+1),[1]*m]], rhs=[2])
    c.linear_constraints.add(lin_expr=[[range(1,m+1),list(y)]])
    constraintMat = np.dot(np.diag(y), xmat).T
    c.linear_constraints.add(lin_expr=[[range(1,m+1), list(constraintMat[i])] for i in range(n)])
    c.linear_constraints.add(lin_expr=[[[0, i], [-1, 1]] for i in range(1, m+1)], senses='L'*m)
    c.solve()
    return 2/(c.solution.get_values()[0]*m)


# Calculate beta-CVaR
def calc_cvar(risks, beta):
    m = len(risks)
    if beta >= 1: return np.max(risks)
    indices_sorted = np.argsort(risks)[::-1] # descent order
    eta = np.zeros(m)
    eta[indices_sorted[range( int(np.ceil(m*(1-beta))) )]] = 1.
    eta[indices_sorted[int(np.ceil(m*(1-beta))-1)]] -= np.ceil(m*(1-beta)) - m*(1-beta)
    return np.dot(risks, eta) / (m*(1-beta))


def kernel_matrix(x, kernel):
    if kernel == 'linear':
        return np.dot(x, x.T)
    elif kernel == 'rbf':
        num, dim = x.shape
        tmp = np.dot(np.ones([num, 1]),
                     np.array([np.linalg.norm(x, axis=1)])**2)
        return np.exp(-(tmp - 2 * np.dot(x, x.T) + tmp.T))


# Uniform distribution on sphere
def runif_sphere(radius, dim, size=1):
    outliers = []
    for i in xrange(size):
        v = np.random.normal(size=dim)
        v = radius * v / np.linalg.norm(v)
        outliers.append(v)
    return np.array(outliers)


if __name__ == '__main__':
    x = np.array([[1,2], [3,4], [5,6]])
    x = np.ones([2000, 10])
    kernel = 'rbf'
    t1 = time.time()
    #print kernel_matrix(x, kernel)
    t2 = time.time()
    print pairwise_kernels(x, metric='rbf', gamma=1.)
    t3 = time.time()
    print 'ORIGINAL:', t2-t1
    print 'scikit-learn:', t3-t2

    outliers = runif_sphere(radius=50, dim=100, size=10)
