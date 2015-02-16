# -*- coding: utf-8 -*-
# Module for standard SVMs

# Set path to CPLEX module
import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # for Ubuntu 64bit
sys.path.append('C:\Program Files (x86)\IBM\ILOG\CPLEX_Studio125\cplex\python\\x86_win32') # for Windows 32bit

# Import libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cplex
import time
from sklearn.metrics import pairwise_kernels

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

##### Training C-SVM using dual #####
def csvm_dual(x, y, cost, kernel, gamma=1., coef0=0., degree=2):
    eps = 1e-5
    m, n = x.shape
    ##### Compute kernel gram matrix #####
    if kernel == 'linear': kmat = pairwise_kernels(x, metric='linear')
    elif kernel == 'rbf': kmat = pairwise_kernels(x, x, metric='rbf', gamma=gamma)
    elif kernel == 'polynomial': kmat = pairwise_kernels(x, metric='polynomial', coef0=coef0, degree=degree)
    else: print 'Undefined Kernel!!'
    qmat = (kmat.T * y).T * y + 1e-8*np.eye(m)
    ##### CPLEX object #####
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set variables #####
    c.variables.add(obj=[-1]*m, ub=[cost]*m)
    ##### Set linear constraint #####
    c.linear_constraints.add(lin_expr=[[range(m), list(y)]], senses='E')
    ##### Set quadratic objective #####
    c.objective.set_quadratic([[range(m), list(qmat[i])] for i in range(m)])
    ##### Select QP method #####
    c.parameters.qpmethod.set(0)
    ##### Solve QP #####
    c.solve()
    alpha = np.array(c.solution.get_values())
    ##### Compute bias #####
    ind_sv = np.array([j for j in xrange(m) if eps <= alpha[j]/cost])
    ind_mv = np.array([j for j in ind_sv if alpha[j]/cost <= 1-eps])
    b = (1 - np.dot(qmat[ind_mv][:,ind_sv], alpha[ind_sv])) * y[ind_mv]
    print 'BIAS:', np.round(b, 3)
    return c, np.mean(b), ind_sv, ind_mv

##### Training C-SVM using primal #####
def csvm_primal(x, y, cost, qpmethod=0):
    num, dim = x.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set names of variables #####
    w_names  = ['w%s' % i for i in range(dim)]
    xi_names = ['xi%s' % i for i in range(num)]
    ##### Set variables (w, b, xi) #####
    c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
    c.variables.add(names=['b'], lb=[- cplex.infinity])
    c.variables.add(obj=[cost]*num, names=xi_names)
    ##### Set quadratic objective #####
    c.objective.set_quadratic_coefficients(zip(range(dim), range(dim), [1]*dim))
    ##### Set linear constraint w*y_i*x_i + b*y_i + xi_i >= 1 for all i #####
    linexpr = [[range(dim+1)+[dim+1+i], list(x[i]*y[i])+[y[i]]+[1.]] for i in range(num)]
    c.linear_constraints.add(lin_expr=linexpr, senses='G'*num, rhs=[1]*num)
    ##### Solve QP #####
    c.parameters.qpmethod.set(qpmethod)
    c.solve()
    return c

##### Convex case of Enu-SVM #####
def solve_convex_primal(xmat, y, nu):
    m, n = xmat.shape
    w_names = ['w'+'%s' % i for i in range(n)]
    xi_names = ['xi'+'%s' % i for i in range(m)]
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set variables #####
    c.variables.add(names = ['rho'], lb = [- cplex.infinity], obj = [- nu*m])
    c.variables.add(names = w_names, lb = [- cplex.infinity] * n)
    c.variables.add(names = ['b'], lb = [- cplex.infinity])
    c.variables.add(names = xi_names, obj = [1.] * m)
    ##### Set quadratic constraint #####
    qexpr = [range(1,n+1), range(1,n+1), [1]*n]
    c.quadratic_constraints.add(quad_expr=qexpr, rhs=1, sense='L', name='norm')
    ##### Set linear constraints #####
    # w * y_i * x_i + b * y_i + xi_i - rho >= 0
    for i in xrange(m):
        c.linear_constraints.add(names = ['margin'+'%s' % i], senses = 'G',
                                 lin_expr = [[w_names + ['b'] + ['xi'+'%s' % i] + ['rho'], list(xmat[i]*y[i]) + [y[i], 1., -1]]])
    # Solve QCLP
    c.solve()
    return c

##### Training nu-SVM using primal #####
def nusvm_primal(x, y, nu):
    num, dim = x.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set names of variables #####
    w_names  = ['w%s' % i for i in range(dim)]
    xi_names = ['xi%s' % i for i in range(num)]
    ##### Set variables (w, b, xi, rho) #####
    c.variables.add(names=w_names, lb=[-cplex.infinity]*dim)
    c.variables.add(names=['b'], lb=[- cplex.infinity])
    c.variables.add(obj=[cost]*num, names=xi_names)
    c.variables.add(names=['rho'])
    ##### Set quadratic objective #####
    c.objective.set_quadratic_coefficients(zip(range(dim), range(dim), [1]*dim))
    ##### Set linear constraint w*y_i*x_i + b*y_i + xi_i - rho >= 0 for all i #####
    linexpr = [[range(dim+1)+[dim+1+i], list(x[i]*y[i])+[y[i]]+[1.]] for i in range(num)]
    c.linear_constraints.add(lin_expr=linexpr, senses='G'*num, rhs=[1]*num)
    ##### Solve QP #####
    c.parameters.qpmethod.set(0)
    c.solve()
    return c

# Non-convex case of Enu-SVM
def solve_nonconvex(xmat, y, nu, w_init, update_rule):
    gamma = 0.9
    m, n = xmat.shape
    w_names = ['w'+'%s' % i for i in range(n)]
    xi_names = ['xi'+'%s' % i for i in range(m)]
    # Set initial point
    w_tilde = w_init
    # Cplex object
    c = cplex.Cplex()
    c.set_results_stream(None)
    # Set variables
    c.variables.add(names = ['rho'], lb = [- cplex.infinity], obj = [- nu*m])
    c.variables.add(names = w_names, lb = [- cplex.infinity] * n)
    c.variables.add(names = ['b'], lb = [- cplex.infinity])
    c.variables.add(names = xi_names, obj = [1.] * m)
    # Set linear constraints: w * y_i * x_i + b * y_i + xi_i - rho >= 0
    c.parameters.lpmethod.set(1)
    for i in xrange(m):
        c.linear_constraints.add(names = ['margin%s' % i], senses = 'G',
                                 lin_expr = [[w_names + ['b'] + ['xi'+'%s' % i] + ['rho'], list(xmat[i]*y[i]) + [y[i], 1., -1]]])
    # w_tilde * w = 1
    c.linear_constraints.add(names=['norm'], lin_expr=[[w_names, list(w_tilde)]], senses='E', rhs=[1.])
    # Iteration
    for i in xrange(1000):
        c.solve()
        w = np.array(c.solution.get_values(w_names))
        # Termination
        if np.linalg.norm(w-w_tilde) < 1e-5: return c
        # Update norm constraint
        if update_rule == 'projection':
            w_tilde = w / np.linalg.norm(w)
        elif update_rule == 'lin_comb': w_tilde = gamma*w_tilde + (1-gamma)*w
        else: 'ERROR: Input valid update rule'
        c.linear_constraints.delete('norm')
        c.linear_constraints.add(names = ['norm'], lin_expr = [[w_names, list(w_tilde)]], senses = 'E', rhs = [1.])
    return c

# Training Enu-SVM
def enusvm(x, y, nu, w_init, update_rule='projection'):
    m, n = x.shape
    result_convex = solve_convex_primal(x, y, nu)
    if -1e-5 < result_convex.solution.get_objective_value() < 1e-5:
        print 'Solve Non-Convex Case'
        result_nonconvex = solve_nonconvex(x, y, nu, w_init, update_rule)
        return result_nonconvex
    else:
        print 'Solve Convex Case'
        return result_convex

if __name__ == '__main__':
    # Import modules
    import time
    
    # Dataset
    dataset = np.loadtxt('Dataset/LIBSVM/liver-disorders/liver-disorders_scale.csv', delimiter=',')
    y = dataset[:,0]
    y[np.where(y==2)] = -1.
    x = dataset[:,1:]
    num, dim = x.shape

    # Synthetic data set including outliers
    ## np.random.seed(0)
    ## num_p, num_n = 1000, 1000
    ## dim = 30
    ## mu_p, mu_n = np.ones(dim), -np.ones(dim)
    ## cov_p = np.eye(dim)
    ## cov_n = np.eye(dim)
    ## x_p = np.random.multivariate_normal(mu_p, cov_p, num_p)
    ## x_n = np.random.multivariate_normal(mu_n, cov_n, num_n)
    ## x = np.vstack([x_p, x_n])
    ## y = np.array([1.]*num_p + [-1.]*(num_n))
    ## num, dim = x.shape

    ## # Toy problem 1
    ## x = np.array([[2., 0.], [3., -1.], [0. ,2.], [-1., 3.]])
    ## y = np.array([1., 1., -1., -1.])
    ## num, dim = x.shape

    ## # Toy problem 2
    ## x = np.array([[2., 0.], [3., -1.], [-5, 7], [0. ,2.], [-1., 3.]])
    ## y = np.array([1., 1., 1., -1., -1.])
    ## num, dim = x.shape

    ## # Toy problem 3
    ## x = np.array([[2, 3], [2, 2], [1, 1.5], [1, 2.5], [1, 3]])
    ## y = np. array([1, -1, 1, 1, -1])
    ## num, dim = x.shape

    # Toy problem 4
    ## x = np.array([[2, 3], [2, 2], [1, 2], [1, 3]])
    ## y = np. array([1, -1, 1, -1])
    ## num, dim = x.shape
    
    ## plt.plot(x[:,0], x[:,1], 'x')
    ## plt.grid()
    ## plt.show()

    # Compute nu_min
    #nu_min = calc_nu_min(x, y)

    # Names of variables
    w_names = ['w'+'%s' % i for i in range(dim)]
    xi_names = ['xi'+'%s' % i for i in range(num)]

    # Hyper-parameter
    cost = 10
    # Kernel function
    kernel = 'linear'
    kmat = pairwise_kernels(x, metric='linear')
    qmat = (kmat.T * y).T * y + 1e-8*np.eye(num)
    
    t1 = time.time()
    res_primal = csvm_primal(x, y, cost, qpmethod=0)
    t2 = time.time()
    w_primal = np.array(res_primal.solution.get_values(w_names))
    b_primal = res_primal.solution.get_values('b')
    print 'TIME:', t2 - t1

    t1 = time.time()
    res_dual, b_dual, ind_sv, ind_mv = csvm_dual(x, y, cost, kernel=kernel)
    t2 = time.time()
    alpha = np.array(res_dual.solution.get_values())
    print 'TIME:', t2 - t1

    ## w1 = np.array(res_primal.solution.get_values(w_names))
    ## ## print w1
    ## print 'OBJ VAL:', res_primal.solution.get_objective_value()
    ## print 'OBJ_VAL:', res_dual.solution.get_objective_value()
    ## w2 = np.dot(alpha*y, x)
    ## ## print w2

    ## print 'alpha:', alpha
    ## print 'w:', w1, w2
    ## res_enusvm = enusvm(x, y, nu=0.2, w_init=np.ones(dim))
