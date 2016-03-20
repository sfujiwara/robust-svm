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

print 'hello'

# ER-SVM + DCA
df = pd.read_csv('dca.csv')
gb = df.groupby(['outlier_ratio', 'nu'], as_index=False)
df = gb.aggregate({'comp_time': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['comp_time_mean'], dtype=int)
df_dca = df.iloc[ind]

# ER-SVM + heuristics
df = pd.read_csv('var.csv')
gb = df.groupby(['outlier_ratio', 'nu'], as_index=False)
df = gb.aggregate({'comp_time': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['comp_time_mean'], dtype=int)
df_var = df.iloc[ind]

# Ramp Loss SVM
df = pd.read_csv('ramp.csv')
gb = df.groupby(['outlier_ratio', 'C'], as_index=False)
df = gb.aggregate({'comp_time': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['comp_time_mean'], dtype=int)
df_ramp = df.iloc[ind]

# Enu-SVM
df = pd.read_csv('enusvm.csv')
gb = df.groupby(['outlier_ratio', 'nu'], as_index=False)
df = gb.aggregate({'comp_time': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['comp_time_mean'], dtype=int)
df_enu = df.iloc[ind]

# Enu-SVM
df = pd.read_csv('csvm.csv')
gb = df.groupby(['outlier_ratio', 'C'], as_index=False)
df = gb.aggregate({'comp_time': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['comp_time_mean'], dtype=int)
df_csvm = df.iloc[ind]

# Set parameters
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.shadow'] = False
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['lines.markeredgewidth'] = 2
plt.rcParams['lines.markersize'] = 8

elw = 2
cs = 3

outlier_ratio = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1])
plt.errorbar(outlier_ratio-0.001, df_dca['comp_time_mean'], yerr=df_dca['comp_time_std'], label='ER-SVM (DCA)', elinewidth=elw, capsize=cs, fmt='-')
plt.errorbar(outlier_ratio-0.002, df_var['comp_time_mean'], yerr=df_var['comp_time_std'], label='ER-SVM (heuristics)', elinewidth=elw, capsize=cs, fmt='--')
plt.errorbar(outlier_ratio+0.002, df_enu['comp_time_mean'], yerr=df_csvm['comp_time_std'], label='Enu-SVM', elinewidth=elw, capsize=cs, fmt='-.')
plt.errorbar(outlier_ratio+0.001, df_csvm['comp_time_mean'], yerr=df_csvm['comp_time_std'], label='C-SVM', elinewidth=elw, capsize=cs, fmt=':')
plt.errorbar(outlier_ratio, df_ramp['comp_time_mean'], yerr=df_ramp['comp_time_std'], label='Ramp', elinewidth=elw, capsize=cs, fmt='-x')
plt.xlabel('Outlier Ratio')
plt.ylabel('training time (sec)')
plt.xlim([-0.01, 0.11])
plt.ylim([-0.1, 2])
plt.legend(loc='upper right')
plt.grid()
plt.show()
