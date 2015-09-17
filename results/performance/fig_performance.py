# -*- coding: utf-8 -*-

import numpy as np
## import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    ## Directory name of result csv
    ## dir_name = "diabetes/"
    ## dir_name = 'liver/'
    ## dir_name = 'heart/'
    dir_name = 'splice/'
    dir_name = 'adult/'
    dir_name = 'vehicle/'
    ## dir_name = 'satimage/'
    ## dir_name = 'svmguide1/'
    dir_name = 'cod-rna/'

    ## Load result csv
    df_dca = pd.read_csv(dir_name+'dca.csv')
    df_var = pd.read_csv(dir_name+'var.csv')
    df_ramp = pd.read_csv(dir_name+'ramp.csv')
    df_enu = pd.read_csv(dir_name+'enu.csv')
    df_libsvm = pd.read_csv(dir_name+'libsvm.csv')
    df_conv = pd.read_csv(dir_name+"conv.csv")

    ## Indices achieving maximum validation performance in each trial
    ind_dca2 = df_dca.groupby(['ratio', 'trial', 'mu']).agg(np.argmax)[["val-acc", "val-f"]]
    ind_dca = df_dca.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]
    ind_var = df_var.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]
    ind_enu = df_enu.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]
    #ind_ramp = df_ramp.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]
    #ind_conv = df_conv.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]
    ## ind_libsvm = df_libsvm.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]

    ## DataFrame for accuracy
    ## ER-SVM + DCA
    tmp = df_dca.iloc[np.array(ind_dca["val-acc"], dtype=int)]
    tmp2 = df_dca.iloc[np.array(ind_dca2["val-acc"], dtype=int)]
    df_acc_dca2 = tmp2.groupby(['ratio', 'mu']).agg({'test-acc': [np.mean, np.std], 'tr-CVaR': [np.min, np.max]})
    df_acc_dca = tmp.groupby(['ratio']).agg({'test-acc': [np.mean, np.std], 'tr-CVaR': [np.min, np.max]})
    ## ER-SVM + heuristics
    tmp = df_var.iloc[np.array(ind_var["val-acc"], dtype=int)]
    df_acc_var = tmp.groupby('ratio').agg({'test-acc': [np.mean, np.std], 'is_convex': [np.min, np.max]})
    ## Enu-SVM
    tmp = df_enu.iloc[np.array(ind_enu["val-acc"], dtype=int)]
    df_acc_enu = tmp.groupby('ratio').agg({'test-acc': [np.mean, np.std], 'is_convex': [np.min, np.max]})
    ## Ramp-loss SVM
    #tmp = df_ramp.iloc[np.array(ind_ramp["val-acc"], dtype=int)]
    df_acc_ramp = tmp.groupby('ratio').agg({'test-acc': [np.mean, np.std]})
    ## ER-SVM + DCA (limited to convex range)
    #tmp = df_conv.iloc[np.array(ind_conv["val-acc"], dtype=int)]
    #df_acc_conv = tmp.groupby('ratio').agg({'test-acc': [np.mean, np.std], 'tr-CVaR': [np.min, np.max]})

    ## DataFrame for f-measure
    ## ER-SVM + DCA
    tmp = df_dca.iloc[np.array(ind_dca["val-f"], dtype=int)]
    tmp2 = df_dca.iloc[np.array(ind_dca2["val-f"], dtype=int)]
    df_f_dca = tmp.groupby(['ratio']).agg({'test-f': [np.mean, np.std], 'tr-CVaR': [np.min, np.max]})
    df_f_dca2 = tmp2.groupby(['ratio', 'mu']).agg({'test-f': [np.mean, np.std], 'tr-CVaR': [np.min, np.max]})
    ## ER-SVM + heuristics
    tmp = df_var.iloc[np.array(ind_var["val-f"], dtype=int)]
    df_f_var = tmp.groupby('ratio').agg({'test-f': [np.mean, np.std], 'is_convex': [np.min, np.max]})
    ## Enu-SVM
    tmp = df_enu.iloc[np.array(ind_enu["val-f"], dtype=int)]
    df_f_enu = tmp.groupby('ratio').agg({'test-f': [np.mean, np.std], 'is_convex': [np.min, np.max]})
    ## Ramp-loss SVM
    #tmp = df_ramp.iloc[np.array(ind_ramp["val-f"], dtype=int)]
    df_f_ramp = tmp.groupby('ratio').agg({'test-f': [np.mean, np.std]})
    ## ER-SVM + DCA (limited to convex range)
    #tmp = df_conv.iloc[np.array(ind_conv["val-f"], dtype=int)]
    #df_f_conv = tmp.groupby('ratio').agg({'test-f': [np.mean, np.std], 'tr-CVaR': [np.min, np.max]})



    ## ratio = np.array([0, 0.03, 0.05, 0.1, 0.15])

    ## Set parameters
    ## plt.rcParams['axes.labelsize'] = 24
    ## plt.rcParams['lines.linewidth'] = 5
    ## plt.rcParams['legend.fontsize'] = 16
    ## plt.rcParams['legend.shadow'] = False
    ## plt.rcParams['xtick.labelsize'] = 12
    ## plt.rcParams['ytick.labelsize'] = 12

    ## plt.plot(ratio, df_gb_dca.groupby('ratio')[test_measure].mean(), label='ER-SVM (DCA)')
    ## ## plt.plot(ratio, df_gb_libsvm.groupby('ratio')[test_measure].mean(), label='C-SVM')
    ## plt.plot(ratio, df_gb_var.groupby('ratio')[test_measure].mean(), label='ER-SVM (heuristics)')
    ## plt.plot(ratio, df_gb_ramp.groupby('ratio')[test_measure].mean(), label='Ramp-SVM')
    ## plt.plot(ratio, df_gb_enu.groupby('ratio')[test_measure].mean(), label='Enu-SVM')
    ## plt.plot(ratio, df_gb_conv.groupby('ratio')[test_measure].mean(), label='ER-SVM (convex range)')
    ## plt.ylabel('Performance')
    ## plt.xlabel('Outlier Ratio')
    ## # plt.ylim([0.55, 0.8])
    ## plt.legend()
    ## plt.grid()
    ## plt.show()
