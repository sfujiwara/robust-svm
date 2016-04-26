# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# Load result csv
df_ersvm = pd.read_csv("results/internet_ad/ersvm.csv")
df_var = pd.read_csv("results/internet_ad/var.csv")
df_enusvm = pd.read_csv("results/internet_ad/enusvm.csv")
df_ramp = pd.read_csv("results/internet_ad/ramp.csv")

# Indices achieving maximum validation performance in each trial
ind_dca = df_ersvm.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]
ind_var = df_var.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]
ind_enu = df_enusvm.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]
ind_ramp = df_ramp.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]

tmp = df_ramp.iloc[np.array(ind_ramp["val-acc"], dtype=int)]
df_acc_ramp = tmp.groupby(['ratio']).agg({'test-acc': [np.mean, np.std]})
tmp = df_ramp.iloc[np.array(ind_ramp["val-f"], dtype=int)]
df_f_ramp = tmp.groupby(['ratio']).agg({'test-f': [np.mean, np.std]})

# DataFrame for accuracy
# ER-SVM + DCA
tmp = df_ersvm.iloc[np.array(ind_dca["val-acc"], dtype=int)]
df_acc_dca = tmp.groupby(['ratio']).agg({'test-acc': [np.mean, np.std], 'tr-CVaR': [np.min, np.max]})
# ER-SVM + heuristics
tmp = df_var.iloc[np.array(ind_var["val-acc"], dtype=int)]
df_acc_var = tmp.groupby('ratio').agg({'test-acc': [np.mean, np.std], 'is_convex': [np.min, np.max]})
# Enu-SVM
tmp = df_enusvm.iloc[np.array(ind_enu["val-acc"], dtype=int)]
df_acc_enu = tmp.groupby('ratio').agg({'test-acc': [np.mean, np.std], 'is_convex': [np.min, np.max]})

# DataFrame for f-measure
# ER-SVM + DCA
tmp = df_ersvm.iloc[np.array(ind_dca["val-f"], dtype=int)]
df_f_dca = tmp.groupby(['ratio']).agg({'test-f': [np.mean, np.std], 'tr-CVaR': [np.min, np.max]})
# ER-SVM + heuristics
tmp = df_var.iloc[np.array(ind_var["val-f"], dtype=int)]
df_f_var = tmp.groupby('ratio').agg({'test-f': [np.mean, np.std], 'is_convex': [np.min, np.max]})
# Enu-SVM
tmp = df_enusvm.iloc[np.array(ind_enu["val-f"], dtype=int)]
df_f_enu = tmp.groupby('ratio').agg({'test-f': [np.mean, np.std], 'is_convex': [np.min, np.max]})
