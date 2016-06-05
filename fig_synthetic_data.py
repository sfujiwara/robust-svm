# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default="results/synthetic")
args = parser.parse_args()

DIR = args.dir + "/"


def flatten_hierarchical_col(col, sep='_'):
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


def percentile_(file_name, param_name):
    df = pd.read_csv(DIR+file_name)
    df_result = df.groupby(['outlier_ratio', param_name], as_index=True).mean()
    df_result["percentile25"] = df.groupby(
        ['outlier_ratio', param_name], as_index=True
    ).quantile(0.25)["test_accuracy"]
    df_result["percentile75"] = df.groupby(
        ['outlier_ratio', param_name], as_index=True
    ).quantile(0.75)["test_accuracy"]
    df_result = df_result.reset_index(inplace=False)
    ind = np.array(df_result.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy'], dtype=int)
    df_result = df_result.iloc[ind][[param_name, "test_accuracy", "percentile25", "percentile75"]]
    return df_result


df_dca = percentile_("dca.csv", "nu")
df_csvm = percentile_("csvm.csv", "C")
df_enu = percentile_("enusvm.csv", "nu")
df_var = percentile_("var.csv", "nu")
df_ramp = percentile_("ramp.csv", "C")

# # ER-SVM + DCA
# df = pd.read_csv(DIR+'dca.csv')
# gb = df.groupby(['outlier_ratio', 'nu'], as_index=False)
# df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
# df.columns = df.columns.map(flatten_hierarchical_col)
# gb = df.groupby(['outlier_ratio'])
# ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
# df_dca = df.iloc[ind]
#
# # ER-SVM + heuristics
# df = pd.read_csv(DIR+'var.csv')
# gb = df.groupby(['outlier_ratio', 'nu'], as_index=False)
# df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
# df.columns = df.columns.map(flatten_hierarchical_col)
# gb = df.groupby(['outlier_ratio'])
# ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
# df_var = df.iloc[ind]
#
# # Ramp Loss SVM
# df = pd.read_csv(DIR+'ramp.csv')
# gb = df.groupby(['outlier_ratio', 'C'], as_index=False)
# df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
# df.columns = df.columns.map(flatten_hierarchical_col)
# gb = df.groupby(['outlier_ratio'])
# ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
# df_ramp = df.iloc[ind]
#
# # Enu-SVM
# df = pd.read_csv(DIR+'enusvm.csv')
# gb = df.groupby(['outlier_ratio', 'nu'], as_index=False)
# df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
# df.columns = df.columns.map(flatten_hierarchical_col)
# gb = df.groupby(['outlier_ratio'])
# ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
# df_enu = df.iloc[ind]
#
# # C-SVM
# df = pd.read_csv(DIR+'csvm.csv')
# gb = df.groupby(['outlier_ratio', 'C'], as_index=False)
# df = gb.aggregate({'test_accuracy': [np.mean, np.std]})
# df.columns = df.columns.map(flatten_hierarchical_col)
# gb = df.groupby(['outlier_ratio'])
# ind = np.array(df.groupby(['outlier_ratio']).agg(np.argmax)['test_accuracy_mean'], dtype=int)
# df_csvm = df.iloc[ind]

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


def plot_(df, label, fmt, d):
    outlier_ratio = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1]) + d
    yerr = np.array([
        np.array(df["test_accuracy"]-df["percentile25"]),
        np.array(df["percentile75"]-df["test_accuracy"]),
    ])
    plt.errorbar(
        outlier_ratio,
        np.array(df["test_accuracy"]),
        yerr=yerr,
        label=label, elinewidth=elw, capsize=cs, fmt=fmt
    )

plot_(df_dca, "ER-SVM (DCA)", "-", -0.001)
plot_(df_var, "ER-SVM (heuristics)", "--", -0.002)
plot_(df_enu, "Enu-SVM", "-.", +0.002)
plot_(df_csvm, "C-SVM", ":", +0.001)
plot_(df_ramp, "Ramp", "-x", 0)

# outlier_ratio = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1])
# plt.errorbar(
#     outlier_ratio-0.001,
#     np.array(df_dca['test_accuracy_mean']),
#     yerr=np.array(df_dca['test_accuracy_std']),
#     label='ER-SVM (DCA)', elinewidth=elw, capsize=cs, fmt='-'
# )
# plt.errorbar(
#     outlier_ratio-0.002,
#     np.array(df_var['test_accuracy_mean']),
#     yerr=np.array(df_var['test_accuracy_std']),
#     label='ER-SVM (heuristics)', elinewidth=elw, capsize=cs, fmt='--'
# )
# plt.errorbar(
#     outlier_ratio+0.002,
#     np.array(df_enu['test_accuracy_mean']),
#     yerr=np.array(df_csvm['test_accuracy_std']),
#     label='Enu-SVM', elinewidth=elw, capsize=cs, fmt='-.'
# )
# plt.errorbar(
#     outlier_ratio+0.001,
#     np.array(df_csvm['test_accuracy_mean']),
#     yerr=np.array(df_csvm['test_accuracy_std']),
#     label='C-SVM', elinewidth=elw, capsize=cs, fmt=':'
# )
# plt.errorbar(
#     outlier_ratio,
#     np.array(df_ramp['test_accuracy_mean']),
#     yerr=np.array(df_ramp['test_accuracy_std']),
#     label='Ramp', elinewidth=elw, capsize=cs, fmt='-x'
# )
plt.xlabel('Outlier Ratio')
plt.ylabel('Test Accuracy')
plt.xlim([-0.01, 0.11])
plt.legend(loc='lower left')
plt.grid()
plt.show()
