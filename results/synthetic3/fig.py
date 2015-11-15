# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def flatten_hierarchical_col(col,sep = '_'):
    if not type(col) is tuple:
        return col
    else:
        new_col = ''
        for leveli,level in enumerate(col):
            if not level == '':
                if not leveli == 0:
                    new_col += sep
                new_col += level
        return new_col

#df.columns = df.columns.map(flattenHierarchicalCol)

print 'hello'

## ER-SVM + DCA
df  = pd.read_csv('dca.csv')
gb = df.groupby(['outlier_ratio', 'nu'], as_index=False)
df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
df_dca = df.iloc[ind]

## ER-SVM + heuristics
df  = pd.read_csv('var.csv')
gb = df.groupby(['outlier_ratio', 'nu'], as_index=False)
df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
df_var = df.iloc[ind]

## Ramp Loss SVM
df = pd.read_csv('ramp.csv')
gb = df.groupby(['outlier_ratio', 'C'], as_index=False)
df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
df_ramp = df.iloc[ind]

## Enu-SVM
df = pd.read_csv('enu.csv')
gb = df.groupby(['outlier_ratio', 'nu'], as_index=False)
df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
df_enu = df.iloc[ind]

## Enu-SVM
df = pd.read_csv('csvm.csv')
gb = df.groupby(['outlier_ratio', 'C'], as_index=False)
df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
df.columns = df.columns.map(flatten_hierarchical_col)
gb = df.groupby(['outlier_ratio'])
ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
df_csvm = df.iloc[ind]

## Set parameters
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

outlier_ratio = np.array([0.0, 0.03, 0.05, 0.1, 0.15, 0.2])
plt.errorbar(outlier_ratio-0.01, df_dca['test_accuracy_mean'], yerr=df_dca['test_accuracy_std'], label='ER-SVM (DCA)', elinewidth=elw, capsize=cs, fmt='-')
plt.errorbar(outlier_ratio-0.005, df_var['test_accuracy_mean'], yerr=df_var['test_accuracy_std'], label='ER-SVM (heuristics)', elinewidth=elw, capsize=cs, fmt='--')
plt.errorbar(outlier_ratio+0.005, df_enu['test_accuracy_mean'], yerr=df_csvm['test_accuracy_std'], label='Enu-SVM', elinewidth=elw, capsize=cs, fmt='-.')
plt.errorbar(outlier_ratio+0.01, df_csvm['test_accuracy_mean'], yerr=df_csvm['test_accuracy_std'], label='C-SVM', elinewidth=elw, capsize=cs, fmt=':')
plt.errorbar(outlier_ratio, df_ramp['test_accuracy_mean'], yerr=df_ramp['test_accuracy_std'], label='Ramp', elinewidth=elw, capsize=cs, fmt='-x')
plt.xlabel('Outlier Ratio')
plt.ylabel('Test Accuracy')
plt.legend(loc='lower left')
plt.grid()
plt.show()
