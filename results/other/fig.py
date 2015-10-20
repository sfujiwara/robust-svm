import numpy as np
import matplotlib.pyplot as plt

t = np.loadtxt('time_thinkpad.csv')
nu = np.loadtxt('nu.csv')

# Set parameters
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['legend.shadow'] = False
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

plt.plot(nu, t/100, label='time')
plt.axvline(x=0.65, color='r', ls=':', label=r'lower threshold')
plt.xlabel('nu')
plt.ylabel('Time (sec)')
plt.ylim(0.04, 0.45)
plt.legend()
plt.grid()
plt.show()
