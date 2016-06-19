
# -*- coding: utf-8 -*-

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

DIR = "results/synthetic/time/"


# In[3]:

# ER-SVM + DCA
df_dca = pd.read_csv(DIR+'dca.csv')

df_dca_nu01 = pd.DataFrame({
    "mean": df_dca[df_dca["nu"]==0.1].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_dca[df_dca["nu"]==0.1].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_dca[df_dca["nu"]==0.1].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)

df_dca_nu05 = pd.DataFrame({
    "mean": df_dca[df_dca["nu"]==0.5].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_dca[df_dca["nu"]==0.5].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_dca[df_dca["nu"]==0.5].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)


# In[4]:

df_dca_nu01


# In[5]:

df_dca_nu05


# In[6]:

# C-SVM
df_csvm = pd.read_csv(DIR+'csvm.csv')

df_csvm_c1 = pd.DataFrame({
    "mean": df_csvm[df_csvm["C"]==1].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_csvm[df_csvm["C"]==1].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_csvm[df_csvm["C"]==1].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)

df_csvm_c25 = pd.DataFrame({
    "mean": df_csvm[df_csvm["C"]==25].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_csvm[df_csvm["C"]==25].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_csvm[df_csvm["C"]==25].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)


# In[7]:

df_csvm_c1


# In[8]:

df_csvm_c25


# In[9]:

# Enu-SVM
df_enusvm = pd.read_csv(DIR+'enusvm.csv')

df_enusvm_nu01 = pd.DataFrame({
    "mean": df_enusvm[df_enusvm["nu"]==0.1].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_enusvm[df_enusvm["nu"]==0.1].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_enusvm[df_enusvm["nu"]==0.1].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)

df_enusvm_nu05 = pd.DataFrame({
    "mean": df_enusvm[df_enusvm["nu"]==0.5].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_enusvm[df_enusvm["nu"]==0.5].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_enusvm[df_enusvm["nu"]==0.5].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)


# In[10]:

# Heuristics
df_var = pd.read_csv(DIR+'var.csv')

df_var_nu01 = pd.DataFrame({
    "mean": df_var[df_var["nu"]==0.1].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_var[df_var["nu"]==0.1].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_var[df_var["nu"]==0.1].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)

df_var_nu05 = pd.DataFrame({
    "mean": df_var[df_var["nu"]==0.5].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_var[df_var["nu"]==0.5].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_var[df_var["nu"]==0.5].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)


# In[11]:

df_var_nu01


# In[12]:

df_var_nu05


# In[13]:

# Ramp Loss SVM
df_ramp = pd.read_csv(DIR+'ramp.csv')

df_ramp_c1 = pd.DataFrame({
    "mean": df_ramp[df_ramp["C"]==1].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_ramp[df_ramp["C"]==1].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_ramp[df_ramp["C"]==1].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)

df_ramp_c25 = pd.DataFrame({
    "mean": df_ramp[df_ramp["C"]==25].groupby(["outlier_ratio"]).mean()["comp_time"],
    "25percentile": df_ramp[df_ramp["C"]==25].groupby(["outlier_ratio"]).quantile(0.25)["comp_time"],
    "75percentile": df_ramp[df_ramp["C"]==25].groupby(["outlier_ratio"]).quantile(0.75)["comp_time"]
}).reset_index(inplace=False)


# In[14]:

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


# In[15]:

# ER-SVM + DCA (nu = 0.1)
yerr_u = np.array(df_dca_nu01["75percentile"] - df_dca_nu01["mean"])
yerr_l = np.array(df_dca_nu01["mean"] - df_dca_nu01["25percentile"])

plt.errorbar(
    df_dca_nu01["outlier_ratio"],
    df_dca_nu01["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='ER-SVM (nu = 0.1)', elinewidth=elw, capsize=cs, fmt='-'
)


# In[16]:

# ER-SVM + DCA (nu = 0.5)
yerr_u = np.array(df_dca_nu05["75percentile"] - df_dca_nu05["mean"])
yerr_l = np.array(df_dca_nu05["mean"] - df_dca_nu05["25percentile"])

plt.errorbar(
    df_dca_nu05["outlier_ratio"] - 0.001,
    df_dca_nu05["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='ER-SVM (nu = 0.5)', elinewidth=elw, capsize=cs, fmt='-^'
)


# In[17]:

# C-SVM (C = 1)
yerr_u = np.array(df_csvm_c1["75percentile"] - df_csvm_c1["mean"])
yerr_l = np.array(df_csvm_c1["mean"] - df_csvm_c1["25percentile"])

plt.errorbar(
    df_csvm_c1["outlier_ratio"],
    df_csvm_c1["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='C-SVM (C = 1)', elinewidth=elw, capsize=cs, fmt=':'
)


# In[18]:

# C-SVM (C = 25)
yerr_u = np.array(df_csvm_c25["75percentile"] - df_csvm_c25["mean"])
yerr_l = np.array(df_csvm_c25["mean"] - df_csvm_c25["25percentile"])

plt.errorbar(
    df_csvm_c25["outlier_ratio"]+0.001,
    df_csvm_c25["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='C-SVM (C = 25)', elinewidth=elw, capsize=cs, fmt=':^'
)


# In[19]:

# Enu-SVM (nu = 0.1)
yerr_u = np.array(df_enusvm_nu01["75percentile"] - df_enusvm_nu01["mean"])
yerr_l = np.array(df_enusvm_nu01["mean"] - df_enusvm_nu01["25percentile"])

plt.errorbar(
    df_enusvm_nu01["outlier_ratio"],
    df_enusvm_nu01["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='Enu-SVM (nu = 0.1)', elinewidth=elw, capsize=cs, fmt='-.'
)


# In[20]:

# Enu-SVM (nu = 0.5)
yerr_u = np.array(df_enusvm_nu05["75percentile"] - df_enusvm_nu05["mean"])
yerr_l = np.array(df_enusvm_nu05["mean"] - df_enusvm_nu05["25percentile"])

plt.errorbar(
    df_enusvm_nu05["outlier_ratio"] + 0.001,
    df_enusvm_nu05["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='Enu-SVM (nu = 0.5)', elinewidth=elw, capsize=cs, fmt='-.^'
)


# In[21]:

# Heuristics (nu = 0.1)
yerr_u = np.array(df_var_nu01["75percentile"] - df_var_nu01["mean"])
yerr_l = np.array(df_var_nu01["mean"] - df_var_nu01["25percentile"])

plt.errorbar(
    df_var_nu01["outlier_ratio"] + 0.001,
    df_var_nu01["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='Heuristics (nu = 0.1)', elinewidth=elw, capsize=cs, fmt='--'
)


# In[22]:

# Heuristics (nu = 0.5)
yerr_u = np.array(df_var_nu05["75percentile"] - df_var_nu05["mean"])
yerr_l = np.array(df_var_nu05["mean"] - df_var_nu05["25percentile"])

plt.errorbar(
    df_var_nu05["outlier_ratio"] - 0.001,
    df_var_nu05["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='Heuristics (nu = 0.5)', elinewidth=elw, capsize=cs, fmt='--^'
)


# In[23]:

# Ramp Loss SVM (C = 1)
yerr_u = np.array(df_ramp_c1["75percentile"] - df_ramp_c1["mean"])
yerr_l = np.array(df_ramp_c1["mean"] - df_ramp_c1["25percentile"])

plt.errorbar(
    df_ramp_c1["outlier_ratio"],
    df_ramp_c1["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='Ramp (C = 1)', elinewidth=elw, capsize=cs, fmt='-s'
)


# In[24]:

# Ramp Loss SVM (C = 25)
yerr_u = np.array(df_ramp_c25["75percentile"] - df_ramp_c25["mean"])
yerr_l = np.array(df_ramp_c25["mean"] - df_ramp_c25["25percentile"])

plt.errorbar(
    df_ramp_c25["outlier_ratio"] + 0.002,
    df_ramp_c25["mean"],
    yerr=np.array([yerr_l, yerr_u]),
    label='Ramp (C = 25)', elinewidth=elw, capsize=cs, fmt='-s'
)


# In[25]:

plt.xlabel('Outlier Ratio')
plt.ylabel('Training Time (sec)')
plt.xlim([-0.005, 0.105])
plt.ylim([-0.01, 0.33])
plt.legend(loc='upper left')
plt.grid()
plt.show()


# In[ ]:



