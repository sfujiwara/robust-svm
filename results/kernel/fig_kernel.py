# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## Set parameters
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.shadow'] = False
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


df_pol = pd.read_csv('kernel_pol_ol5.csv')
df_lin = pd.read_csv('kernel_lin_ol5.csv')

res_pol = df_pol.groupby(['nu'], as_index=False).mean()
res_lin = df_lin.groupby(['nu'], as_index=False).mean()

m_pol = np.array(df_pol.groupby(['nu']).obj_val.median() > 0)
m_lin = np.array(df_lin.groupby(['nu']).obj_val.median() > 0)
nu = np.array(res_pol['nu'])

plt.plot(nu, 1 - res_pol['test_error'], label='Polynomial')
plt.plot(nu, 1 - res_lin['test_error'], '--', label='Linear')
plt.plot(nu[m_pol], 1 - res_pol['test_error'].ix[m_pol], 'bs', ms=9, markeredgecolor='none')
plt.plot(nu[m_lin], 1 - res_lin['test_error'].ix[m_lin], 'gs', ms=9, markeredgecolor='none')
plt.grid()
plt.legend(loc='lower right')
plt.xlabel('nu')
plt.ylabel('Test Accuracy')
plt.xlim(0.07, 0.83)
plt.show()

## plt.plot(nu, df_lin.groupby(['nu']).comp_time.mean(), label='Linear')
## #plt.plot(nu, df_lin.groupby(['nu']).obj_val.max())
## plt.plot(nu, df_pol.groupby(['nu']).comp_time.mean(), label='Polynomial')
## #plt.plot(nu, df_pol.groupby(['nu']).obj_val.max())
## #plt.ylim(-0.1, 0.2)
## plt.legend()
## plt.grid()
## plt.show()
