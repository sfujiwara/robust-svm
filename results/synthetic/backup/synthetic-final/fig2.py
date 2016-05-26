# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def flatten_hierarchical_col(col,sep = '_'):
    if not type(col) is tuple:
        return col
    else:
        new_col = ''
        for leveli, level in enumerate(col):
            if not level == '':
                if not leveli == 0:
                    new_col += sep
                new_col += level
        return new_col

# df.columns = df.columns.map(flattenHierarchicalCol)


# ER-SVM + DCA
df_dca = pd.read_csv('dca.csv')
gb = df_dca.groupby(['nu', 'outlier_ratio'], as_index=False)
# gb = df_dca.groupby(['outlier_ratio'], as_index=False)
df_dca = gb.aggregate({'comp_time': [np.mean, np.std]})

# ER-SVM + heuristics
df = pd.read_csv('var.csv')
gb = df.groupby(['nu', 'outlier_ratio'], as_index=False)
# gb = df.groupby(['outlier_ratio'], as_index=False)
df_var = gb.aggregate({'comp_time': [np.mean, np.std]})

# Ramp Loss SVM
df = pd.read_csv('ramp.csv')
gb = df.groupby(['C', 'outlier_ratio'], as_index=False)
# gb = df.groupby(['outlier_ratio'], as_index=False)
df_ramp = gb.aggregate({'comp_time': [np.mean, np.std]})

# Enu-SVM
df = pd.read_csv('enusvm.csv')
gb = df.groupby(['nu', 'outlier_ratio'], as_index=False)
# gb = df.groupby(['outlier_ratio'], as_index=False)
df_enu = gb.aggregate({'comp_time': [np.mean, np.std]})

# C-SVM
df = pd.read_csv('csvm.csv')
gb = df.groupby(['C', 'outlier_ratio'], as_index=False)
# gb = df.groupby(['outlier_ratio'], as_index=False)
df_csvm = gb.aggregate({'comp_time': [np.mean, np.std]})

# Set parameters
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.shadow'] = False
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['lines.markeredgewidth'] = 1
plt.rcParams['lines.markersize'] = 9

elw = 0
cs = 3

x = range(9)
outlier_ratio = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1])
x = outlier_ratio
df = df_dca[df_dca['nu'] == 0.1]
plt.errorbar(x, df['comp_time']['mean'],  yerr=df['comp_time']['std'],  label='ER-SVM (nu = 0.1)', elinewidth=elw, capsize=cs, fmt='-')
df = df_dca[df_dca['nu'] == 0.5]
plt.errorbar(x, df['comp_time']['mean'],  yerr=df['comp_time']['std'],  label='ER-SVM (nu = 0.5)', elinewidth=elw, capsize=cs, fmt='-^')
# C-SVM
df = df_csvm[df_csvm['C'] == 1e0]
plt.errorbar(x, df['comp_time']['mean'], yerr=df['comp_time']['std'], label='C-SVM (C = 1)', elinewidth=elw, capsize=cs, fmt=':')
df = df_csvm[df_csvm['C'] == 25]
plt.errorbar(x, df['comp_time']['mean'], yerr=df['comp_time']['std'], label='C-SVM (C = 25)', elinewidth=elw, capsize=cs, fmt=':^')
# Enu-SVM
df = df_enu[df_enu['nu'] == 0.1]
plt.errorbar(x, df['comp_time']['mean'],  yerr=df['comp_time']['std'],  label='Enu-SVM (nu = 0.1)', elinewidth=elw, capsize=cs, fmt='-.')
df = df_enu[df_enu['nu'] == 0.5]
plt.errorbar(x, df['comp_time']['mean'],  yerr=df['comp_time']['std'],  label='Enu-SVM (nu = 0.5)', elinewidth=elw, capsize=cs, fmt='-.^')
# Heuristics
df = df_var[df_var['nu'] == 0.1]
plt.errorbar(x, df['comp_time']['mean'],  yerr=df['comp_time']['std'],  label='Heuristics (nu = 0.1)', elinewidth=elw, capsize=cs, fmt='--')
df = df_var[df_var['nu'] == 0.5]
plt.errorbar(x, df['comp_time']['mean'],  yerr=df['comp_time']['std'],  label='Heuristics (nu = 0.5)', elinewidth=elw, capsize=cs, fmt='--^')
# Ramp SVM
df = df_ramp[df_ramp['C'] == 1e0]
plt.errorbar(x, df['comp_time']['mean'], yerr=df['comp_time']['std'], label='Ramp (C = 1)', elinewidth=elw, capsize=cs, fmt='-s')
df = df_ramp[df_ramp['C'] == 25]
plt.errorbar(x, df['comp_time']['mean'], yerr=df['comp_time']['std'], label='Ramp (C = 25)', elinewidth=elw, capsize=cs, fmt='-s')
# plt.xlabel('Outlier Ratio')
plt.xlabel('Outlier Ratio')
plt.ylabel('Training Time (sec)')
plt.xlim([-0.005, 0.105])
plt.ylim([-0.05, 0.3])
plt.legend(loc='upper left')
plt.grid()
plt.show()
