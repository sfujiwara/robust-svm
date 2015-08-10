# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print "hello"

if __name__ == '__main__':
    measure = "accuracy"
    # measure = "f-measure"
    pd.set_option('line_width', 200)
    pd.set_option("display.max_rows", 200)
    if measure == "accuracy":
        val_measure = 'val-acc'
        test_measure = 'test-acc'
    elif measure == "f-measure":
        val_measure + "val-f"
        test_measure = "test-f"
    # Load result csv
    dir_name = 'liver-disorders/'
    #dir_name = "diabetes/"
    dir_name = 'liver/'
    #dir_name = 'heart2/'
    df_dca = pd.read_csv(dir_name+'dca.csv')
    df_var = pd.read_csv(dir_name+'var.csv')
    df_ramp = pd.read_csv(dir_name+'ramp.csv')
    df_enu = pd.read_csv(dir_name+'enu.csv')
    df_libsvm = pd.read_csv(dir_name+'libsvm.csv')
    df_conv = pd.read_csv(dir_name+"conv.csv")

    ## Indices achieving maximum validation performance in each trial
    ind_dca = df_dca.groupby(['ratio', 'trial']).agg(np.argmax)[["val-acc", "val-f"]]
    test = df_dca.iloc[np.array(ind_dca["val-acc"], dtype=int)]

    gb_dca = df_dca.groupby(['ratio', 'trial'])
    gb_libsvm = df_libsvm.groupby(['ratio', 'trial'])
    gb_var = df_var.groupby(['ratio', 'trial'])
    gb_ramp = df_ramp.groupby(['ratio', 'trial'])
    gb_enu = df_enu.groupby(['ratio', 'trial'])
    gb_conv = df_conv.groupby(['ratio', 'trial'])

    df_gb_dca = df_dca[gb_dca[val_measure].transform(max) == df_dca[val_measure]].groupby(['ratio', 'trial'], as_index=False).first()
    df_gb_var = df_var[gb_var[val_measure].transform(max) == df_var[val_measure]].groupby(['ratio', 'trial'], as_index=False).first()
    df_gb_enu = df_enu[gb_enu[val_measure].transform(max) == df_enu[val_measure]].groupby(['ratio', 'trial'], as_index=False).first()
    df_gb_ramp = df_ramp[gb_ramp[val_measure].transform(max) == df_ramp[val_measure]].groupby(['ratio', 'trial'], as_index=False).first()
    df_gb_libsvm = df_libsvm[gb_libsvm[val_measure].transform(max) == df_libsvm[val_measure]].groupby(['ratio', 'trial'], as_index=False).first()
    df_gb_conv = df_conv[gb_conv[val_measure].transform(max) == df_conv[val_measure]].groupby(['ratio', 'trial'], as_index=False).first()

    ratio = np.array([0, 0.03, 0.05, 0.1, 0.15])

    # Set parameters
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['lines.linewidth'] = 5
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    plt.plot(ratio, df_gb_dca.groupby('ratio')[test_measure].mean(), label='ER-SVM (DCA)')
    plt.plot(ratio, df_gb_libsvm.groupby('ratio')[test_measure].mean(), label='C-SVM')
    plt.plot(ratio, df_gb_var.groupby('ratio')[test_measure].mean(), label='ER-SVM (heuristics)')
    plt.plot(ratio, df_gb_ramp.groupby('ratio')[test_measure].mean(), label='Ramp-SVM')
    plt.plot(ratio, df_gb_enu.groupby('ratio')[test_measure].mean(), label='Enu-SVM')
    plt.plot(ratio, df_gb_conv.groupby('ratio')[test_measure].mean(), label='ER-SVM (convex range)')
    plt.ylabel('Performance')
    plt.xlabel('Outlier Ratio')
    # plt.ylim([0.55, 0.8])
    plt.legend()
    plt.grid()
    plt.show()

    res_dca = df_gb_dca.groupby('ratio')[test_measure].mean()
    res_libsvm = df_gb_libsvm.groupby('ratio')[test_measure].mean()
    res_var = df_gb_var.groupby('ratio')[test_measure].mean()
    res_ramp = df_gb_ramp.groupby('ratio')[test_measure].mean()

