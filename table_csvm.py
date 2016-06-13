# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str)
args = parser.parse_args()

DATASET_NAME = args.dir

# Load result csv
df_csvm = pd.read_csv("{}/csvm.csv".format(DATASET_NAME))

# Indices achieving maximum validation performance in each trial
ind_csvm = df_csvm.groupby(["ratio", "trial"]).agg(np.argmax)[["val-acc", "val-f"]]

# DataFrame for accuracy

# C-SVM
tmp = df_csvm.iloc[np.array(ind_csvm["val-acc"], dtype=int)]
df_acc_csvm = tmp.groupby("ratio").agg({'test-acc': [np.mean, np.std]})

# DataFrame for f-measure

# C-SVM
tmp = df_csvm.iloc[np.array(ind_csvm["val-f"], dtype=int)]
df_f_csvm = tmp.groupby("ratio").agg({'test-f': [np.mean, np.std]})
