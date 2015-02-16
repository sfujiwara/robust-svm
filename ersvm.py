## -*- coding: utf-8 -*-

import sys
## Ubuntu
sys.path.append('/opt/ibm/ILOG/CPLEX_Studio126/cplex/python/x86-64_linux')
## Windows
## sys.path.append('C:\Program Files (x86)\IBM\ILOG\CPLEX_Studio125\cplex\python\\x86_win32')
import numpy as np
import matplotlib.pyplot as plt
import cplex
import time
import enusvm

##### Minimize a difference of CVaR by DCA (using linear kernel) #####
def diff_cvar(dmat, labels, weight, bias, nu, mu):
    max_itr = 100
    num, dim = dmat.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    w_names = ['w' +'%s' % i for i in range(dim)]
    xi_names = ['xi'+'%s' % i for i in range(num)]
    coeffs_t = []
    ##### Initialize risk #####
    risks = - labels * (np.dot(dmat, weight) + bias)
    ##### Initialize eta #####
    eta = calc_eta(risks, mu)
    eta_bef = calc_eta(risks, mu)
    ##### Initialize t #####
    obj_val_bef = num * (calc_cvar(risks, 1-nu)*nu - calc_cvar(risks, 1-mu)*mu)
    t = max(0, obj_val_bef / 0.99)
    coeffs_t.append(t)
    print 't:\t\t', t
    ##### Set variables and objective function #####
    c.variables.add(names=w_names, lb=[-cplex.infinity]*dim, ub=[cplex.infinity]*dim)
    c.variables.add(names=['b'], lb=[-cplex.infinity], ub=[cplex.infinity])
    c.variables.add(names=xi_names, obj=[1.]*num, lb=[0.]*num, ub=[cplex.infinity]*num)
    c.variables.add(names=['alpha'], obj=[nu*num], lb=[-cplex.infinity], ub=[cplex.infinity])
    ##### Set quadratic constraint #####
    c.quadratic_constraints.add(name='norm', quad_expr=[w_names, w_names, [1.]*dim], rhs=1., sense='L')
    ##### Set linear constraints w*y_i*x_i + b*y_i + xi_i - alf >= 0 #####
    linexpr = [[w_names+['b', 'xi%s' % i, 'alpha'], list(dmat[i]*labels[i])+[labels[i], 1., 1.]] for i in range(num)]
    names = ['margin%s' % i for i in range(num)]
    c.linear_constraints.add(names=names, senses='G'*num, lin_expr=linexpr)
    ## for i in xrange(m):
    ##     linexpr = [[w_names + ['b'] + ['xi%s' % i] + ['alpha'], list(dmat[i]*labels[i]) + [labels[i]] + [1.] + [1.]]]
    ##     c.linear_constraints.add(names = ['margin%s' % i], senses = 'G', lin_expr = linexpr)
    ##### Set QP optimization method #####
    c.parameters.qpmethod.set(0)
    ##### Iteration #####
    for i in xrange(max_itr):
        print '\nITERATION:\t', i+1
        print '|w|:\t\t', np.linalg.norm(weight)
        ##### Update objective function #####
        c.objective.set_linear('b', np.dot(1-eta, labels))
        c.objective.set_linear(zip(w_names, np.dot(labels*(1-eta), dmat) - 2*t*weight))        
        ##### Solve subproblem #####
        c.solve()
        weight = np.array(c.solution.get_values(w_names))
        xi = np.array(c.solution.get_values(xi_names))
        bias = c.solution.get_values('b')
        alpha = c.solution.get_values('alpha')
        ##### Update risk #####
        risks = - labels * (np.dot(dmat, weight) + bias)
        ##### Update eta #####
        eta = calc_eta(risks, mu)
        ##### Objective Value #####
        obj_val = num * (calc_cvar(risks, 1-nu) * nu - calc_cvar(risks, 1-mu) * mu)
        ##### Update t #####
        t = max(1e-5 + obj_val/0.999, 0.)
        coeffs_t.append(t)
        print 'OBJCTIVE VALUE:\t', obj_val
        print 't:\t\t', t
        ##### Termination #####
        diff_obj = np.abs((obj_val_bef - obj_val))
        print 'DIFF_OBJ:\t', diff_obj
        if diff_obj < 1e-10: break
        ## if np.all(eta == eta_bef): break
        obj_val_bef = obj_val
        eta_bef = eta
    print 'OVER MAXIMUM ITERATION'
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

# Calculate beta-CVaR
def calc_cvar(risks, beta):
    m = len(risks)
    if beta >= 1: return np.max(risks)
    indices_sorted = np.argsort(risks)[::-1] # descent order
    eta = np.zeros(m)
    eta[indices_sorted[range( int(np.ceil(m*(1-beta))) )]] = 1.
    eta[indices_sorted[int(np.ceil(m*(1-beta))-1)]] -= np.ceil(m*(1-beta)) - m*(1-beta)
    return np.dot(risks, eta) / (m*(1-beta))

# Calculate eta
def calc_eta(risks, mu):
    m = len(risks)
    indices_sorted = np.argsort(risks)[::-1]
    eta = np.zeros(m)
    eta[indices_sorted[range(int(np.ceil(m*mu)))]] = 1.
    eta[indices_sorted[int(np.ceil(m*mu)-1)]] -= np.ceil(m*mu) - m*mu
    eta = 1 - eta
    return eta

# Non-linear classification using kernel
# Minimize a difference of CVaR by DCA (using non-linear kernel)
def diff_cvar_kernel(dmat, labels, a, bias, nu, mu, kernel):
    print 'Iteration:\t', 0
    m, n = dmat.shape
    c = cplex.Cplex()
    c.set_results_stream(None)
    a_names  = ['a%s'  % i for i in range(m)]
    xi_names = ['xi%s' % i for i in range(m)]
    kmat = kernel_matrix(dmat, kernel)
    # Initialize risk
    risks = - labels * (np.dot(kmat, a) + bias)
    # Initialize eta
    eta = calc_eta(risks, mu)
    # Initialize t
    obj_val_old = calc_cvar(risks, 1-nu) * nu - calc_cvar(risks, 1-mu) * mu
    t = max(0, obj_val_old/0.95+1e-5)
    # Set variables and objective function
    c.variables.add(names = a_names, lb = [-cplex.infinity]*m, ub = [cplex.infinity]*m)
    c.variables.add(names = ['b'], lb = [-cplex.infinity], ub = [cplex.infinity])
    c.variables.add(names = xi_names, obj = [1./m]*m, lb = [0.]*m, ub = [cplex.infinity]*m)
    c.variables.add(names = ['alpha'], obj = [nu], lb = [-cplex.infinity], ub = [cplex.infinity])
    # Set quadratic constraints
    # ||w||^2 <= 1 where w = Sum{a_j*Phi(x_j)} <--> aKa <= 1

    kmat = kmat + np.eye(m)*1e-8 # Make kernel matrix be positive semidefinite
    ind1 = []
    print 'Bottole neck?'
    for i in range(m): ind1 = ind1 + ['a%s' % i] * m
    print 'End bottle neck?'
    ind2 = ['a%s' % i for i in range(m)] * m
    val  = list(np.reshape(kmat, m*m))
    print 'begin to set quadratic constraint'
    t1 = time.time()
    c.quadratic_constraints.add(name = 'norm', quad_expr = [ind1, ind2, val], rhs = 1., sense = 'L')
    t2 = time.time()
    c.quadratic_constraints.add(name = 'norm', quad_expr = cplex.SparseTriple(ind1, ind2, val), rhs = 1., sense = 'L')
    t3 = time.time()
    print t2-t1, t3-t2
    print 'finished to set quadratic constraint'
    # Set linear constraints
    # y_i(wx_i + b) + xi_i + alpha >= 0 where w = Sum{a_j*Phi(x_j)}
    print 'Begin to set linear constraints'
    for i in xrange(m):
        names_variables = a_names + ['b'] + ['xi%s' % i] + ['alpha']
        #coeffs = [labels[i] * kernel(dmat[i], dmat[j]) for j in range(m)] + [labels[i]] + [1.] + [1.]
        coeffs = [labels[i] * kmat[i,j] for j in range(m)] + [labels[i]] + [1.] + [1.]
        c.linear_constraints.add(names = ['margin%s' % i], senses = 'G', lin_expr = [[names_variables, coeffs]])
    print 'Finished to set linear constraints'
    # Save the problem as text file
    c.write('test.lp')
    # Iteration
    for i in xrange(20):
        print '\nIteration:\t', i+1
        #print '|w|^2 = aKa:\t', np.dot(np.dot(a, kmat), a)
        # Update objective function
        # - Sum_i{1 - eta_i^k} * y_i * b
        print 'Begin to update objective for b'
        c.objective.set_linear('b', np.dot(1-eta, labels) / m)
        print 'Finished to update objective'
        # w_k * w <--> a_k^T K a, 
        #coeffs1 = np.array([sum((1-eta[k]) * labels[k] * kernel(dmat[j], dmat[k]) for k in range(m)) for j in range(m)]) / m
        print 'Begin to update objective for w'
        coeffs1 = np.array([sum((1-eta[k]) * labels[k] * kmat[j,k] for k in range(m)) for j in range(m)]) / m
        # w_k * w <--> a_k^T K a, 
        coeffs2 = - 2 * t * np.dot(a, kmat)
        c.objective.set_linear(zip(a_names, coeffs1+coeffs2))
        print 'Finished to update objective'
        # Solve subproblem
        print 'start'
        c.solve()
        print 'end'
        a     = np.array(c.solution.get_values(a_names))
        xi    = np.array(c.solution.get_values(xi_names))
        bias  = c.solution.get_values('b')
        alpha = c.solution.get_values('alpha')
        # Calculate risk: r_i(w_k, b_k)
        risks = - np.array([labels[j] * (sum(a[k] * kmat[j,k] for k in range(m)) + bias) for j in range(m)])
        # Update eta
        eta = calc_eta(risks, mu)
        # Update t
        print 'Begin to update t'
        obj_val = calc_cvar(risks, 1-nu) * nu - calc_cvar(risks, 1-mu) * mu
        print 'Finished to update t'
        t = max(0., obj_val/0.9)
        print 't:\t\t', t
        # Termination
        obj_val_new = calc_cvar(risks, 1-nu) * nu - calc_cvar(risks, 1-mu) * mu
        diff_obj = np.abs((obj_val_old - obj_val_new) / obj_val_old)
        print 'DIFF_OBJ:\t', diff_obj
        if diff_obj < 1e-5: break
        obj_val_old = obj_val_new
        #print 'WEIGHT:\t\t', np.round(np.dot(dmat.T, a), 3)
    return c, eta

##### Extended Robust SVM (Dual) #####
def ersvm_dual(dmat, y, nu, mu, kernel, gamma=1., coef0=0., degree=2):
    ##### Constant values #####
    EPS = 1e-5
    MAX_ITR = 25
    NUM, DIM = dmat.shape
    ##### Initial point #####
    eta = np.ones(NUM)
    ##### Compute kernel gram matrix #####
    if kernel == 'linear':
        kmat = pairwise_kernels(x, metric='linear')
    elif kernel == 'rbf':
        kmat = pairwise_kernels(x, x, metric='rbf', gamma=gamma)
    elif kernel == 'polynomial':
        kmat = pairwise_kernels(x, metric='polynomial', coef0=coef0, degree=degree)
    else:
        print 'Undefined Kernel!!'
    #kmat = kmat + 1e-8*np.eye(m)
    qmat = (kmat.T * y).T * y + 1e-8*np.eye(NUM)
    qmat = np.round(qmat, 10)
    ##### CPLEX object #####
    c = cplex.Cplex()
    c.set_results_stream(None)
    ##### Set variables #####
    c.variables.add(obj=[0]*NUM)
    ##### Set quadratic objective #####
    ## qmat_sparse = [cplex.SparsePair(ind=range(m), val=list(qmat[i])) for i in range(m)]
    ## c.objective.set_quadratic(qmat_sparse)
    print 'Set quadratic objective'
    c.objective.set_quadratic([[range(NUM), list(qmat[i])] for i in range(NUM)])
    ##### Set linear constraint #####
    c.linear_constraints.add(lin_expr=[[range(NUM), list(y)]], senses='E', rhs=[0])
    c.linear_constraints.add(lin_expr=[[range(NUM), [1]*NUM]], senses='E', rhs=[nu-mu])
    ##### Set QP optimization method #####
    c.parameters.qpmethod.set(1)
    for i in xrange(MAX_ITR):
        ##### Update constraints #####
        c.variables.set_lower_bounds(zip(range(NUM), list((eta-1)/NUM)))
        c.variables.set_upper_bounds(zip(range(NUM), list(eta/NUM)))
        ##### Update linear objective #####
        linexpr = np.dot(qmat, alpha_bef) * 2
        c.objective.set_linear(lin_expr=linexpr)
        ##### Solve subproblem #####
        c.solve()
        print c.solution.status[c.solution.get_status()]
        print 'OBJ_VAL:', c.solution.get_objective_value()
        alpha = np.array(c.solution.get_values())
        ##### Compute bias, rho, and decision values #####
        ind_mv_p = [j for j in xrange(NUM) if EPS <= (alpha[j]*NUM) <= 1-EPS and y[j] > 1-EPS]
        ind_mv_n = [j for j in xrange(NUM) if EPS <= (alpha[j]*NUM) <= 1-EPS and y[j] < -1+EPS]
        wx = np.dot(alpha*y, kmat)
        print wx[ind_mv_p], wx[ind_mv_n]
        #print max(alpha), min(alpha)
        bias = -(np.mean(wx[ind_mv_p])+np.mean(wx[ind_mv_n])) / 2
        print 'BIAS =', bias
        #rho = 
        dv = wx + bias
        ##### Update eta #####
        #
        risks = - dv * y
        eta_new = np.zeros(NUM)
        ind_sorted = np.argsort(risks)[::-1]
        print bias
        eta_new[ind_sorted[range(int(np.ceil(NUM*mu)))]] = 1.
        eta_new[ind_sorted[int(np.ceil(NUM*mu)-1)]] -= np.ceil(NUM*mu) - NUM*mu
        eta_new = 1 - eta_new
        #print eta_new
        #
        if all(eta == eta_new):
            print 'CONVERGED: TOTAL_ITR =', i+1
            return c, eta
        else: eta = eta_new
    print 'OVER MAXIMUM ITERATION'
    return c, eta

def prox_decomp(x, y, nu, mu):
    NUM, DIM = x.shape
    MAX_ITR = 30
    c = cplex.Cplex()
    ##### Names of variables #####
    w_names  = ['w' +'%s' % i for i in range(DIM)]
    xi_names = ['xi'+'%s' % i for i in range(NUM)]
    eta_names = ['eta%s' % i for i in range(NUM)]
    ##### Set variables and linear objective #####
    c.variables.add(names=w_names, lb=[-cplex.infinity]*DIM, ub=[cplex.infinity]*DIM)
    c.variables.add(names=['b'], lb=[-cplex.infinity], ub=[cplex.infinity])
    c.variables.add(names=xi_names, obj=[1./(nu-mu)/NUM]*NUM, lb=[0.]*NUM, ub=[cplex.infinity]*NUM)
    c.variables.add(names=['rho'], obj=[-1.], lb=[-cplex.infinity], ub=[cplex.infinity])
    c.variables.add(names=eta_names, ub=[1.]*NUM)
    ##### Set linear constraints #####
    ##### Set quadratic objective #####
    ind1 = eta_names + eta_names + xi_names + xi_names
    ind2 = eta_names + xi_names + eta_names + xi_names
    c.objective.set_quadratic_coefficients(zip(ind1, ind2, [1.]*NUM*4))
    return c
    
def heuristic_dr(xmat, y, nu, w_init, gamma=0.1, heuristics=False):
    max_itr = 100
    m, n = xmat.shape
    w = np.zeros(n)
    # w_bef = np.zeros(NUM)
    # b_bef = 0.
    b = 0
    active_set = np.arange(m)
    ind_act = np.arange(m)
    for i in range(max_itr):
        ##### Update nu #####
        nu_i = (nu * (1-gamma)**i * m) / len(active_set)
        x_active = xmat[active_set]
        y_active = y[active_set]
        ##### Check bounded or not
        nu_max = enusvm.calc_nu_max(y_active)
        if nu_i > nu_max:
            print 'nu_i:', nu_i, '>', 'nu_max', nu_max
            break
        ##### Solve subproblem if bounded
        result = enusvm.enusvm(xmat[active_set], y[active_set], nu_i, w_init)
        w_new = np.array(result.solution.get_values(range(1, n+1)))
        b = result.solution.get_values(n+1)
        ##### Heuristic termination (1e-4 or 1e-5 is better) #####
        if heuristics and np.abs(1 - np.dot(w, w_new)) < 1e-4:
            print 'Heuristic Termination:', 1-np.dot(w, w_new)
            break
        else: w = w_new
        ##### Update loss and active set #####
        loss = - (np.dot(xmat, w) + b) * y
        card_active = np.ceil(m * (1 - nu + nu*(1-gamma)**(i+1)))
        new_active_set = np.argsort(loss)[range(np.int(card_active))]
        ##### Terminate if active set does not change #####
        if set(active_set) == set(new_active_set):
            print 'VaR Minimization'
            break
        else: active_set = new_active_set
        w_init = w
    print 'ITR:', i + 1
    return [result, w, b, active_set, nu_i]

if __name__ == '__main__':
    import pandas as pd
    # Read a UCI dataset
    ## dataset = np.loadtxt('Dataset/LIBSVM/liver-disorders/liver-disorders_scale.csv', delimiter=',')
    ## dataset = np.loadtxt('Dataset/LIBSVM/heart/heart_scale.csv', delimiter=',')
    dataset = np.loadtxt('Dataset/LIBSVM/liver-disorders/liver-disorders_scale.csv', delimiter=',')
    ## dataset = np.loadtxt('Dataset/LIBSVM/sonar/sonar_scale.csv', delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape
    ## dmat_test = dmat
    ## labels_test = labels
    
    ##### Set parameters #####
    ## nu_cand = np.arange(0.05, 0.6, 0.03)
    nu_cand = np.arange(0.05, 0.65, 0.05)
    nu_cand = np.array([0.2])
    ##### Names of variables #####
    w_names = ['w%s'  % i for i in range(dim)]
    xi_names = ['xi%s' % i for i in range(num)]

    num_seed = 10
    obj_h = np.zeros([num_seed, len(nu_cand)])
    obj_dc = np.zeros([num_seed, len(nu_cand)])
    mu_cand = np.zeros([num_seed, len(nu_cand)])
    err_h = np.zeros([num_seed, len(nu_cand)])
    err_dc = np.zeros([num_seed, len(nu_cand)])
    for i in range(num_seed):
        ##### Initial point #####
        np.random.seed(i+75+3)
        w_init = np.random.normal(0, 1, dim)
        w_init /= np.linalg.norm(w_init)
        b_init = np.random.normal(0, 1)
        for j in range(len(nu_cand)):
            nu = nu_cand[j]
            ##### Heuristic algorithm #####
            res_h, w_h, b_h, active_set, nu_i = heuristic_dr(x, y, nu, w_init, gamma = 0.03/nu, heuristics = True)
            risks_h = - (np.dot(x, w_h) + b_h) * y
            mu = 1 - len(active_set) / float(num)
            mu_cand[i,j] = mu
            obj_h[i,j] = nu * calc_cvar(risks_h, 1-nu) - mu * calc_cvar(risks_h, 1-mu)
            err_h[i,j] = sum(y * (np.dot(x, w_h) + b_h) <= 0)

            ##### Difference of CVaRs with linear kernel #####
            result, eta = diff_cvar(x, y, w_init, b_init, nu, mu)
            w_dc = np.array(result.solution.get_values(w_names))
            xi     = np.array(result.solution.get_values(xi_names))
            b_dc   = result.solution.get_values('b')
            alpha = result.solution.get_values('alpha')
            risks_dc = - y * (np.dot(x, w_dc) + b_dc)
            obj_dc[i,j] = nu * calc_cvar(risks_dc, 1-nu) - mu * calc_cvar(risks_dc, 1-mu)
            err_dc[i,j] = sum(y * (np.dot(x, w_dc) + b_dc) <= 0)

    obj_diff = (obj_dc - obj_h) / abs(obj_dc)
    diff_ave = np.array([np.mean(i) for i in obj_diff.T])
    diff_max = np.array([np.max(i) for i in obj_diff.T])
    diff_min = np.array([np.min(i) for i in obj_diff.T])
    diff_sd = np.array([np.std(i) for i in obj_diff.T])
    plt.errorbar(nu_cand, diff_ave, yerr=diff_sd, label='Mean')
    #plt.plot(nu_cand, diff_max, label='Max')
    #plt.plot(nu_cand, diff_min, label='Min')
    plt.legend()
    plt.grid()
    plt.xlabel('nu')
    plt.ylabel('[OBJ(DCA) - OBJ(Heuristics)] / |OBJ(DCA)|')
    plt.show()

    np.savetxt('obj_dc.csv', obj_dc, fmt='%0.9f', delimiter=',')
    np.savetxt('obj_h.csv', obj_h, fmt='%0.9f', delimiter=',')
    np.savetxt('err_dc.csv', err_dc, fmt='%0.9f', delimiter=',')
    np.savetxt('err_h.csv', err_h, fmt='%0.9f', delimiter=',')

    ## res, eta = diff_cvar(x, y, w_init, b_init, nu=0.5, mu=0.11)
    ## w_dc = np.array(res.solution.get_values(w_names))
    ## xi     = np.array(res.solution.get_values(xi_names))
    ## b_dc   = res.solution.get_values('b')
    ## alpha = res.solution.get_values('alpha')
    ## r_dc = - y * (np.dot(x, w_dc) + b_dc)
    ## ind_ol = np.where(eta != 1)[0]
    ## print np.sort(r_dc[ind_ol])
    ## print np.sort(r_dc)[305:]
