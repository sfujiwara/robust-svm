
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ## Load results
    sampling_size = np.loadtxt('sampling_size.csv', delimiter=',')
    time_ersvm1 = np.loadtxt('ersvm_nu01.csv', delimiter=',')
    time_ersvm5 = np.loadtxt('ersvm_nu05.csv', delimiter=',')
    time_libsvm0 = np.loadtxt('libsvm_c0.csv', delimiter=',')
    time_libsvm4 = np.loadtxt('libsvm_c4.csv', delimiter=',')
    time_enusvm1 = np.loadtxt('enusvm_nu05.csv', delimiter=',')
    time_enusvm5 = np.loadtxt('enusvm_nu05.csv', delimiter=',')
    
    ## Set parameters for plot
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['lines.markeredgewidth'] = 0
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    elw = 2
    cs = 5

    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_ersvm1],
                 yerr=[np.std(i) for i in time_ersvm1],
                 label='ER-SVM (nu = 0.1)', elinewidth=elw, capsize=cs, marker='o')
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_ersvm5],
                 yerr=[np.std(i) for i in time_ersvm5],
                 label='ER-SVM (nu = 0.5)', elinewidth=elw, capsize=cs, marker='^')
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_libsvm0],
                 yerr=[np.std(i) for i in time_libsvm0],
                 label='LIBSVM (C = 1e0)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_libsvm4],
                 yerr=[np.std(i) for i in time_libsvm4],
                 label='LIBSVM (C = 1e4)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_enusvm1],
                 yerr=[np.std(i) for i in time_enusvm1],
                 label='Enu-SVM (nu = 0.1)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_enusvm5],
                 yerr=[np.std(i) for i in time_enusvm5],
                 label='Enu-SVM (nu = 0.5)', elinewidth=elw, capsize=cs)
    plt.xlabel('# training samples')
    plt.ylabel('training time (sec)')
    plt.legend(loc='upper left')
    plt.ylim([-50, 1500])
    plt.xlim([-1000, 61000])
    plt.grid()
    plt.show()
