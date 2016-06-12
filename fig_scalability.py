
import numpy as np
import matplotlib.pyplot as plt

DIR = "results/scalability/"

# Load results
sampling_size = np.loadtxt(DIR+'sampling_size.csv', delimiter=',')
time_ersvm1 = np.loadtxt(DIR+'ersvm_nu01.csv', delimiter=',')
time_ersvm5 = np.loadtxt(DIR+'ersvm_nu05.csv', delimiter=',')
time_libsvm0 = np.loadtxt(DIR+'libsvm_c1e0.csv', delimiter=',')
time_libsvm4 = np.loadtxt(DIR+'libsvm_c1e3.csv', delimiter=',')
time_enusvm1 = np.loadtxt(DIR+'enusvm_nu01.csv', delimiter=',')
time_enusvm5 = np.loadtxt(DIR+'enusvm_nu05.csv', delimiter=',')
time_var1 = np.loadtxt(DIR+'var_nu01.csv', delimiter=',')
time_var5 = np.loadtxt(DIR+'var_nu05.csv', delimiter=',')

# Set parameters for plot
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 8
plt.rcParams['lines.markeredgewidth'] = 2
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.shadow'] = False
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
elw = 2
cs = 3


def plot_percentile(data, label, fmt, length=8, position=0.):
    yerr = np.vstack([
        np.mean(data, axis=1) - np.percentile(data, 25, axis=1),
        np.percentile(data, 75, axis=1) - np.mean(data, axis=1)
    ])
    plt.errorbar(
        sampling_size[:length] + position,
        np.mean(data, axis=1)[:length], yerr=yerr[:, :length],
        label=label, elinewidth=2, capsize=3, fmt=fmt
    )


plot_percentile(time_ersvm1, "ER-SVM (nu = 0.1)", "-")
plot_percentile(time_ersvm5, "ER-SVM (nu = 0.5)", "-x", position=-500)
plot_percentile(time_libsvm0, "C-SVM (C = 1e0)", ":")
plot_percentile(time_libsvm4, "C-SVM (C = 1e3)", ":x", -3)
plot_percentile(time_enusvm1, "Enu-SVM (nu = 0.1)", "-.", position=500)
plot_percentile(time_enusvm5, "Enu-SVM (nu = 0.5)", "-.x")
plot_percentile(time_var1, "Heuristics (nu = 0.1)", "--", -1)
plot_percentile(time_var5, "Heuristics (nu = 0.5)", "--x", -1)

plt.xlabel('# training samples')
plt.ylabel('training time (sec)')
plt.legend(loc='upper left')
plt.ylim([-50, 1500])
plt.xlim([-9000, 61000])
plt.grid()
plt.show()
