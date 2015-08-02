
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load results
    sampling_size = np.loadtxt('sampling_size.csv', delimiter=',')
    time_ersvm1 = np.loadtxt('ersvm_nu01.csv', delimiter=',')
    time_ersvm5 = np.loadtxt('ersvm_nu05.csv', delimiter=',')
    time_libsvm0 = np.loadtxt('libsvm_c1e0.csv', delimiter=',')
    time_libsvm4 = np.loadtxt('libsvm_c1e3.csv', delimiter=',')
    time_enusvm1 = np.loadtxt('enusvm_nu01.csv', delimiter=',')
    time_enusvm5 = np.loadtxt('enusvm_nu05.csv', delimiter=',')
    time_var1 = np.loadtxt('var_nu01.csv', delimiter=',')
    time_var5 = np.loadtxt('var_nu05.csv', delimiter=',')

    # Set parameters for plot
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['lines.markeredgewidth'] = 0
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    elw = 2
    cs = 5

    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_ersvm1],
                 yerr=[np.std(i) for i in time_ersvm1],
                 label='ER-SVM (nu = 0.1)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_ersvm5],
                 yerr=[np.std(i) for i in time_ersvm5],
                 label='ER-SVM (nu = 0.5)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_libsvm0],
                 yerr=[np.std(i) for i in time_libsvm0],
                 label='LIBSVM (C = 1e0)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size[:-3],
                 [np.mean(i) for i in time_libsvm4[:-3]],
                 yerr=[np.std(i) for i in time_libsvm4[:-3]],
                 label='LIBSVM (C = 1e3)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_enusvm1],
                 yerr=[np.std(i) for i in time_enusvm1],
                 label='Enu-SVM (nu = 0.1)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size,
                 [np.mean(i) for i in time_enusvm5],
                 yerr=[np.std(i) for i in time_enusvm5],
                 label='Enu-SVM (nu = 0.5)', elinewidth=elw, capsize=cs)
    plt.errorbar(sampling_size[:-1],
                 [np.mean(i) for i in time_var1[:-1]],
                 yerr=[np.std(i) for i in time_var1[:-1]],
                 label='Heuristics (nu = 0.1)', elinewidth=elw, capsize=cs, ls='--')
    plt.errorbar(sampling_size[:-1],
                 [np.mean(i) for i in time_var5[:-1]],
                 yerr=[np.std(i) for i in time_var5[:-1]],
                 label='Heuristics (nu = 0.5)', elinewidth=elw, capsize=cs, ls='--')
    plt.xlabel('# training samples')
    plt.ylabel('training time (sec)')
    plt.legend(loc='upper left')
    plt.ylim([-50, 1500])
    plt.xlim([-9000, 61000])
    plt.grid()
    plt.show()
