# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Set parameters
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.shadow'] = False
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['lines.markeredgewidth'] = 2

df_pol10 = pd.read_csv('kernel_pol_ol10.csv')
df_pol00 = pd.read_csv('kernel_pol.csv')
df_lin10 = pd.read_csv('kernel_lin_ol10.csv')
df_lin00 = pd.read_csv('kernel_lin.csv')

res_pol00 = df_pol00.groupby(['nu'], as_index=False).mean()
res_lin00 = df_lin00.groupby(['nu'], as_index=False).mean()
res_pol10 = df_pol10.groupby(['nu'], as_index=False).mean()
res_lin10 = df_lin10.groupby(['nu'], as_index=False).mean()

m_pol00 = np.array(df_pol00.groupby(['nu']).obj_val.max() > 0)
m_lin00 = np.array(df_lin00.groupby(['nu']).obj_val.max() > 0)
m_pol10 = np.array(df_pol10.groupby(['nu']).obj_val.max() > 0)
m_lin10 = np.array(df_lin10.groupby(['nu']).obj_val.max() > 0)
nu = np.array(res_pol00['nu'])

p0 = plt.plot(nu, 1 - res_pol00['test_error'], label='Polynomial (0 %)')
l0 = plt.plot(nu, 1 - res_lin00['test_error'], label='Linear (0 %)')
p10 = plt.plot(nu, 1 - res_pol10['test_error'], '--', label='Polynomial (10 %)')
l10 = plt.plot(nu, 1 - res_lin10['test_error'], '--', label='Linear (10 %)')

plt.plot(nu[m_pol00], np.array(1 - res_pol00['test_error'].ix[m_pol00]), 'bo', ms=9)  # , markeredgecolor='none')
plt.plot(nu[m_lin00], np.array(1 - res_lin00['test_error'].ix[m_lin00]), 'g^', ms=9)  # , markeredgecolor='none')
plt.plot(nu[m_pol10], np.array(1 - res_pol10['test_error'].ix[m_pol10]), 'rs', ms=9)  # , markeredgecolor='none')
plt.plot(nu[m_lin10], np.array(1 - res_lin10['test_error'].ix[m_lin10]), 'cd', ms=9)  # , markeredgecolor='none')

plt.grid()
plt.legend(loc='lower right')
plt.xlabel('nu')
plt.ylabel('Test Accuracy')
plt.xlim(0.07, 0.83)
plt.ylim(0.54, 0.71)
plt.show()
