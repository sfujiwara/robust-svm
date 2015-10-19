## -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Plot Figure
    params = {#'backend': 'ps',  
              'axes.labelsize': 24,
              #'text.fontsize': 18,
              #'legend.fontsize': 28,
              'legend.fontsize': 24,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              #'text.usetex': False,
              }
    plt.rcParams.update(params)

    nu_cand = np.arange(0.05, 0.65, 0.05)
    obj_h = np.loadtxt('obj_h.csv', delimiter=',')
    obj_dc = np.loadtxt('obj_dc.csv', delimiter=',')
    obj_diff = (obj_dc - obj_h) / abs(obj_dc)
    #obj_diff = (obj_dc - obj_h) / abs(obj_h)
    diff_ave = np.array([np.mean(i) for i in obj_diff.T])
    diff_max = np.array([np.max(i) for i in obj_diff.T])
    diff_min = np.array([np.min(i) for i in obj_diff.T])
    diff_med = np.array([np.median(i) for i in obj_diff.T])
    diff_per75 = np.array([np.percentile(i, 75) for i in obj_diff.T])
    diff_per25 = np.array([np.percentile(i, 25) for i in obj_diff.T])
    diff_sd = np.array([np.std(i) for i in obj_diff.T])
    plt.grid()
    # plt.errorbar(nu_cand, diff_ave, yerr=diff_sd, label='Mean', lw=5, elinewidth=3, capsize=5)
    ## plt.plot(nu_cand, diff_max)
    ## plt.plot(nu_cand, diff_min)
    ## plt.plot(nu_cand, diff_per75, '--', label='first quantile', lw=5)
    ## plt.plot(nu_cand, diff_med, '-', label='median', lw=5)
    ## plt.plot(nu_cand, diff_per25, ':', label='third quantile', lw=5)
    ## plt.plot(data, '-', label='time')
    ## plt.axvline(x=0.65, lw=lw, color='r', ls=':', label=r'$\underline{a}$')
    ## plt.axvline(x=0.65, lw=lw, color='r', ls=':', label=r'lower threshold')
    ## plt.axvline(x=0.335, lw=lw, color='r', ls=':')
    plt.legend(shadow=True, prop={'size': 18}, loc='upper left')
    plt.xlabel(r'$\nu$')
    #plt.ylabel('[OBJ(DCA) - OBJ(Heuristics)] / |OBJ(DCA)|', fontsize=20)
    #plt.ylim(-1e12, 1)
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

    tmp = np.empty(obj_dc.shape)
    eps = 0.03
    for i in range(300):
        for j in range(12):
            if obj_dc[i,j] > 0 and obj_h[i,j] > 0:
                if obj_dc[i,j] < obj_h[i,j] * (1-eps): tmp[i,j] = 0
                elif obj_dc[i,j] > obj_h[i,j] * (1+eps): tmp[i,j] = 1
                else: tmp[i,j] = 2
            elif obj_dc[i,j] < 0 and obj_h[i,j] < 0:
                if obj_dc[i,j] < obj_h[i,j] * (1+eps): tmp[i,j] = 0            
                elif obj_dc[i,j] > obj_h[i,j] * (1-eps): tmp[i,j] = 1
                else: tmp[i,j] = 2
            else:
                if obj_dc[i,j] < obj_h[i,j]: tmp[i,j] = 0            
                elif obj_dc[i,j] > obj_h[i,j] * (1-eps): tmp[i,j] = 1

    win = np.array([sum(i==0) for i in tmp.T])
    lose = np.array([sum(i==1) for i in tmp.T])
    draw = np.array([sum(i==2) for i in tmp.T])
    ##test = obj_dc < obj_h
    #win = [sum(i) for i in test.T]
    plt.bar(nu_cand, win, 0.03, label='win')
    plt.bar(nu_cand, draw, 0.03, win, color='r', label='draw')
    plt.grid()
    plt.xlabel('nu')
    plt.legend(loc='lower right')
    plt.show()
