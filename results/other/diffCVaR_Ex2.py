## -*- coding: utf-8 -*-
import sys
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio125/cplex/python/x86_sles10_4.1') # Ubuntu
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux') # Ubuntu
sys.path.append('C:\Program Files (x86)\IBM\ILOG\CPLEX_Studio125\cplex\python\\x86_win32') # Windows
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cplex
import enusvm
import rsvm
import timeit
import time

# Calculate nu_min
def compute_nu_min(xmat, y):
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

#def calc_nu_max(y):
    

# Minimize a difference of CVaR using DCA
def dca_diff_cvar(dmat, labels, weight, bias, nu, mu):
    m, n = dmat.shape; c = cplex.Cplex(); c.set_results_stream(None)
    w_names = ['w'+'%s' % i for i in range(n)]; xi_names = ['xi'+'%s' % i for i in range(m)]

    # Initialize eta
    eta = np.zeros(m)
    risks = - labels * (np.dot(dmat, weight) + bias)
    indices_sorted = np.argsort(risks)
    eta[indices_sorted[range(int(np.ceil(m*mu)))]] = 1.
    eta[indices_sorted[int(np.ceil(m*mu)-1)]] -= np.ceil(m*mu) - m*mu
    eta = 1 - eta
    #print sum(eta), m*mu
    # Initialize t
    obj_val_old =  cvar(weight, bias, 1-nu, dmat, labels) * nu - cvar(weight, bias, 1-mu, dmat, labels) * mu
    obj_val_sub = cvar(weight, bias, 1-nu, dmat, labels) * nu
    for i in range(m):
        obj_val_sub += (1-eta[i]) * labels[i] * (np.dot(weight, dmat[i]) + bias) / m
    print obj_val_sub, obj_val_old
    t1 = max(0., obj_val_old)
    t2 = max(0., obj_val_sub) / 1.9
    t = max(t1, t2)
    print 't:', t1, t2, t

    # Set variables and objective function
    c.variables.add(names = w_names, lb = [-cplex.infinity]*n, ub = [cplex.infinity]*n)
    c.variables.add(names = ['b'], lb = [-cplex.infinity], ub = [cplex.infinity])
    c.variables.add(names = xi_names, obj = [1./m]*m, lb = [0.]*m, ub = [cplex.infinity]*m)
    c.variables.add(names = ['alpha'], obj = [nu], lb = [-cplex.infinity], ub = [cplex.infinity])

    # Set constraints
    c.quadratic_constraints.add(name = 'norm', quad_expr = [w_names, w_names, [1.]*n], rhs = 1., sense = 'L')
    for i in xrange(m):
        c.linear_constraints.add(names = ['margin'+'%s' % i], senses = 'G',
                                 lin_expr = [[w_names + ['b'] + ['xi'+'%s' % i] + ['alpha'], list(dmat[i]*labels[i]) + [labels[i]] + [1.] + [1.]]])

    for i in xrange(50):
        print 'DCVaR:', weight
        # Update objective function
        c.objective.set_linear('b', np.dot(1-eta,labels)/m)
        c.objective.set_linear(zip(w_names, np.dot(labels*(1-eta), dmat)/m - 2*t*weight))        
        # Solve subproblem
        c.solve()
        weight = np.array(c.solution.get_values(w_names))
        xi = np.array(c.solution.get_values(xi_names))
        bias = c.solution.get_values('b')
        alpha = c.solution.get_values('alpha')
        # Update eta
        eta = np.zeros(m)
        risks = - labels * (np.dot(dmat, weight) + bias)
        indices_sorted = np.argsort(risks)[::-1]
        eta[indices_sorted[range(int(np.ceil(m*mu)))]] = 1.
        eta[indices_sorted[int(np.ceil(m*mu)-1)]] -= np.ceil(m*mu) - m*mu
        eta = 1-eta
        
        #print 'w:', weight
        print 'OBJ_VAL:', np.round((nu-mu)*alpha + np.dot(eta, xi)/m, 5), np.round(cvar(weight, bias, 1-nu, dmat, labels) * nu - cvar(weight, bias, 1-mu, dmat, labels) * mu, 5), np.round(c.solution.get_objective_value() + 2*t, 5)
        print '(1-nu)-CVaR:', cvar(weight, bias, 1-nu, dmat, labels), alpha + sum(xi)/(nu*m)
        
        # Update t
        obj_val_sub = cvar(weight, bias, 1-nu, dmat, labels) * nu
        for i in range(m):
            obj_val_sub += (1-eta[i]) * labels[i] * (np.dot(weight, dmat[i]) + bias) / m
        t1 = max(0., cvar(weight, bias, 1-nu, dmat, labels) * nu - cvar(weight, bias, 1-mu, dmat, labels) * mu)
        t2 = max(0., obj_val_sub) / 1.9
        t = max(t1, t2)
        print 't:', t1, t2, t

        # Termination
        obj_val_new = cvar(weight, bias, 1-nu, dmat, labels) * nu - cvar(weight, bias, 1-mu, dmat, labels) * mu
        diff_obj = np.abs((obj_val_old - obj_val_new) / obj_val_old)
        print 'DIFF_OBJ', diff_obj
        if diff_obj < 1e-5:
            print i
            break
        obj_val_old = obj_val_new
        
    return c, eta

# Calculate beta-CVaR
def cvar(weight, bias, beta, dmat, labels):
    m, n = dmat.shape
    risks = - labels * (np.dot(dmat, weight) + bias)
    if beta >= 1: return np.max(risks)
    indices_sorted = np.argsort(risks)[::-1]
    eta = np.zeros(m)
    eta[indices_sorted[range(int(np.ceil(m*(1-beta))))]] = 1.
    eta[indices_sorted[int(np.ceil(m*(1-beta))-1)]] -= np.ceil(m*(1-beta)) - m*(1-beta)
    return np.dot(risks, eta) / (m*(1-beta))



if __name__ == '__main__':
    ## Read a UCI dataset
    dataset = np.loadtxt('dataset/libsvm/liver_scale.csv', delimiter=',')
    labels = dataset[:,-1]
    dmat = dataset[:,:-1]
    m, n = dmat.shape
    margins = []
    obj_vals = []
    obj_vals2 = []
    comp_time = []
    num_train = 104

    a1a = np.loadtxt('dataset/libsvm/Adult/a1a.csv', delimiter=',')
    dmat_a1a = a1a[:,1:]
    labels_a1a = a1a[:,0]

    ## Prepare the names of variables
    w_names = ['w' + '%s' % i for i in range(n)]; xi_names = ['xi' + '%s' % i for i in range(num_train)]
    
    ## Initial point
    w_init_dcvar = np.ones(n) / np.sqrt(n); b_init_dcvar = 0.0
    w_init_enusvm = np.ones(n) / np.sqrt(n)


    np.random.seed(3)
    indices_random = np.arange(m)
    np.random.shuffle(indices_random)
    dmat_train = dmat[indices_random[:num_train]]
    labels_train = labels[indices_random[:num_train]]
    dmat_test = dmat[indices_random[num_train:]]
    labels_test = labels[indices_random[num_train:]]
    nu_min = compute_nu_min(dmat_train, labels_train)
    nu_max = (num_train - abs(sum(labels_test)))*1. / num_train
    
    # Set parameters
    nu = np.arange(nu_max-0.01, 0.1, -0.01)
    mu = 0.03
    # Error
    errors_dcvar = np.zeros([len(nu)])
    # Experiments using Artificial Dataset
    for i in range(len(nu)):
        # Learn using DiffCVaR
        print 'Difference of CVaR'
        print 'nu:', nu[i]
        #t = timeit.timeit(stmt='diffCVaR_EX2.dca_diff_cvar(dmat_train, labels_train, w_init_dcvar, b_init_dcvar, nu[i], mu)', number=10, 'import diffCVaR_EX2')
        #t1 = time.time()
        comp_time_tmp = []
        for j in range(500):
            t1 = time.time()
            c, eta = dca_diff_cvar(dmat_train, labels_train, w_init_dcvar, b_init_dcvar, nu[i], mu)
            t2 = time.time()
            comp_time_tmp.append(t2-t1)
        #t2 = time.time()
        comp_time.append(comp_time_tmp)
        weight_dcvar = np.array(c.solution.get_values(w_names))
        bias_dcvar = c.solution.get_values('b')
        print 'w:', weight_dcvar
        alpha_dcvar = c.solution.get_values('alpha')
        xi_dcvar = c.solution.get_values(xi_names)
        w_init_dcvar = weight_dcvar / np.linalg.norm(weight_dcvar)
        b_init_dcvar = bias_dcvar
        margins.append(- alpha_dcvar)
        obj_val = (cvar(weight_dcvar, bias_dcvar, 1-nu[i], dmat_train, labels_train) * nu[i] - cvar(weight_dcvar, bias_dcvar, 1-mu, dmat_train, labels_train) * mu)/(nu[i]-mu)
        obj_val2 = alpha_dcvar + np.dot(xi_dcvar, eta)/(num_train*(nu[i]-mu))
        obj_val2 = (nu[i]-mu)*alpha_dcvar + np.dot(eta, xi_dcvar)/num_train
        print obj_val, obj_val2
        
        obj_vals.append(obj_val)
        obj_vals2.append(obj_val2)
        errors_dcvar[i] = sum(labels_test * (np.dot(dmat_test, weight_dcvar) + bias_dcvar) <= 0)

    obj_vals = np.array(obj_vals)
    obj_vals2 = np.array(obj_vals2)
    comp_time = np.array(comp_time)
    comp_time_ave = np.array([np.mean(comp_time[i]) for i in range(len(nu))])
    comp_time_med = np.array([np.median(comp_time[i]) for i in range(len(nu))])
    # Plot Figure
    params = {#'backend': 'ps',  
              'axes.labelsize': 24,
              #'text.fontsize': 18,
              'legend.fontsize': 28,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              #'text.usetex': False,
              }
    plt.rcParams.update(params)
    lw = 5
    ## plt.plot(nu, errors_dcvar/241., '-', lw=lw, label='test error')
    ## plt.xlabel(r'$\nu$', fontsize=24)
    ## plt.ylabel('Test Error', fontsize=24)
    ## plt.axvline(x=0.65, lw=lw, ls=':', color='r', label=r'lower threshold')
    ## #plt.axvline(x=0.335, lw=lw, ls=':', color='r')
    ## plt.legend(shadow=False, prop={'size': 28}, loc='upper left')
    ## plt.ylim(0.25,0.40)
    ## plt.grid()
    ## plt.show()

    ## plt.plot(nu, obj_vals, '-', lw=lw, label='Objctive Value')
    ## plt.plot(nu, margins, '--', lw=lw, label='Margin Value')
    ## plt.axvline(x=0.65, lw=lw, color='r', ls=':')
    ## plt.axvline(x=0.335, lw=lw, color='r', ls=':')
    ## plt.legend(shadow=True, prop={'size': 18}, loc='upper left')
    ## plt.xlabel(r'$\nu$', fontsize=20)
    ## plt.ylabel('Value', fontsize=20)
    ## plt.ylim(-0.16, 0.2)
    ## plt.grid()
    ## plt.show()

    ## from matplotlib import rc
    ## rc('text', usetex=True)
    #comp_time = np.loadtxt('time_thinkpad.csv', delimiter=',')
    plt.plot(nu, comp_time_ave, '-', lw=lw, label='mean')
    plt.plot(nu, comp_time_med, '-', lw=lw, label='median')
    #plt.axvline(x=0.65, lw=lw, color='r', ls=':', label=r'$\underline{a}$')
    plt.axvline(x=0.65, lw=lw, color='r', ls=':', label=r'lower threshold')
    #plt.axvline(x=0.335, lw=lw, color='r', ls=':')
    #plt.legend(shadow=True, prop={'size': 18}, loc='upper left')
    plt.xlabel(r'$\nu$', fontsize=24)
    plt.ylabel('Time (sec)', fontsize=24)
    plt.ylim(0.04, 0.45)
    plt.legend()
    plt.grid()
    plt.show()

    np.savetxt('nu.csv', nu, fmt='%.10f', delimiter=',')
    np.savetxt('time.csv', comp_time, fmt='%.10f', delimiter=',')
