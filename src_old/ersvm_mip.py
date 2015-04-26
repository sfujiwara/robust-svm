# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import sys
# cplexモジュールへpathを通す
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux')
sys.path.append('C:\Program Files (x86)\IBM\ILOG\CPLEX_Studio125\cplex\python\\x86_win32') # Windows
import cplex
import enusvm

def ersvm_mip(xmat, y, nu_I, nu_O):
    m, n = xmat.shape
    bigM = 1e2
    c = cplex.Cplex()
    c.set_results_stream(None)
    # 変数を用意
    c.variables.add(obj = [1], names = ['alpha'], lb = [- cplex.infinity])
    c.variables.add(names = ['w%s' % i for i in range(1,n+1)], lb = [- cplex.infinity] * n)
    c.variables.add(names = ['b'], lb = [- cplex.infinity])
    c.variables.add(obj = [1./((m - np.floor(nu_O * m)) * nu_I)] * m, names = ['xi%s' % i for i in range(1,m+1)])
    c.variables.add(names = ['u%s' % i for i in range(1,m+1)], types = 'B'*m)
    # 制約
    c.quadratic_constraints.add(quad_expr = [range(1,n+1), range(1,n+1), [1]*n], rhs = 1, sense = 'L', name='norm')
    c.linear_constraints.add(lin_expr = [[['u%s' % i for i in range(1,m+1)], [1] * m]], rhs = [nu_O * m], senses = 'L')
    for i in xrange(m):
        linexpr = [[range(n+2) + [n+2+i] + [n+m+2+i], [1] + list(xmat[i]*y[i]) + [y[i]] + [1] + [bigM]]]
        c.linear_constraints.add(lin_expr = linexpr, senses='G')
    c.write('test.lp')
    c.solve()
    return c

if __name__ == '__main__':
    np.random.seed(0) # seedの固定
    #平均
    mu1 = [-1,-1]
    mu2 = [1,1]
    #共分散
    cov = [[1,0],[0,1]]
    #500はデータ数
    num = 30
    x1 = np.random.multivariate_normal(mu1,cov,num)
    x2 = np.random.multivariate_normal(mu2,cov,num)
    xmat = np.vstack([x1, x2])
    y = np.array([1]*num + [-1]*num)
    m, n = xmat.shape
    gamma = 0.2
    nu = 0.6
    nu_I = 0.6
    nu_O = 0.1

    c = ersvm_mip(xmat, y, nu_I=nu_I, nu_O=nu_O)
    
    ## plt.plot(x1[:,0], x1[:,1], 'wo')
    ## plt.plot(x2[:,0], x2[:,1], 'x')
    ## plt.grid()
    ## plt.show()
    ## diff = []
    ## size = []
    ## size2=[]
    ## obj_mip = []
    ## obj_dr = []
    ## for i in range(1):
    ##     x1 = np.random.multivariate_normal(mu1,cov,num)
    ##     x2 = np.random.multivariate_normal(mu2,cov,num)
    ##     xmat = np.vstack([x1, x2])
    ##     y = np.array([1]*num + [-1]*num)
        
    ##     result_h, active_dr, itr = doubly_robust(xmat=xmat, y=y, nu=nu, w_init=0, gamma=gamma, heuristics=True)
    ##     active_dr = set(active_dr)
    ##     inactive_dr = set(range(m)) - active_dr
    ##     print 'Optimal Value:', result_h.solution.get_objective_value()

    ##     nu_I = (nu * (1-gamma)**(itr)) / (1 - nu + nu * (1-gamma)**(itr))
    ##     nu_O = nu - (nu * (1-gamma)**(itr))

    ##     c = ersvm_mip(xmat, y, nu_I=nu_I, nu_O=nu_O)
    ##     uvec = np.array(c.solution.get_values(['u%s' % i for i in range(1,m+1)]))
    ##     inactive_mip = set(np.where(uvec==1)[0])

    ##     diff.append(len(inactive_mip - inactive_dr))
    ##     size.append(len(inactive_mip))
    ##     size2.append(len(inactive_dr))
    ##     obj_dr.append(result_h.solution.get_objective_value())
    ##     obj_mip.append(c.solution.get_objective_value())
        
    ## diff = np.double(np.array(diff))
    ## size = np.double(np.array(size))
    ## obj_dr = np.array(obj_dr)
    ## obj_mip = np.array(obj_mip)
    ## print '---------- RESULT ----------'
    ## ## print 'Status1:', c.solution.status[c.solution.get_status()]
    ## ## print 'Status2:', c.solution.get_status_string()
    ## ## print 'Method:', c.solution.method[c.solution.get_method()]
    ## print 'Primal Feasible:', c.solution.is_primal_feasible()
    ## ## print 'Dual Feasible:', c.solution.is_dual_feasible()
    ## print 'Optimal Value:', c.solution.get_objective_value()
    ## ## print 'Optimal Solution:', c.solution.get_values()
    ## opt_sol = c.solution.get_values()
    ## #print 'Dual Solution:', c.solution.get_dual_values()

    ## print inactive_mip
    ## print inactive_dr

    ## #plt.hist(diff/size, bins=20)
    ## plt.hist(np.abs((obj_dr-obj_mip)/obj_mip), bins=30)
    ## plt.grid()
    ## plt.show()
