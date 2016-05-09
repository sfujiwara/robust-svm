# -*- coding: utf-8 -*-

from sklearn.datasets import load_svmlight_file
from sklearn.datasets.mldata import fetch_mldata
from sklearn import svm
import numpy as np
import pandas as pd


def load_data(name):
    if name == "mushrooms":
        return load_mushrooms()
    if name == "gisette":
        return load_gisette()
    if name == "usps":
        return load_usps()
    if name == "dna":
        return load_dna()
    if name == "internet_ad":
        return load_internet_ad()
    if name == "w6a":
        return load_w6a()
    if name == "madelon":
        return load_madelon()
    if name == "connect-4":
        return load_connect4()
    if name == "aloi":
        return load_aloi()


def load_mushrooms():
    data = fetch_mldata("mushrooms")
    x = data["data"].todense()
    y = data["target"].astype(float)
    y[y > 1.5] = -1.
    return x, y, None


def load_gisette():
    x, y = load_svmlight_file("data/libsvm/gisette/gisette_scale.bz2")
    x = x.toarray()
    return x, y, None


# def load_protein():
#     x, y = load_svmlight_file("data/libsvm/protein/protein.bz2", dtype=str)
#     # x = x.toarray()
#     return x, y, None


# def load_usps():
#     x, y = load_svmlight_file("data/libsvm/usps/usps.bz2")
#     x = x.toarray()
#     x_outlier = x[y == 2]
#     ind = (y != 2)
#     x, y = x[ind], y[ind]
#     y[y != 1] = -1.
#     clf = svm.SVC(kernel="linear", cache_size=2000)
#     clf.fit(x, y)
#     y_outlier = clf.predict(x_outlier) * -1
#     return x, y, x_outlier, y_outlier


def load_usps():
    x, y = load_svmlight_file("data/libsvm/usps/usps.bz2")
    x = x.todense()
    ind_ol = (y == 9)
    ind = (y != 9)
    ind_p = y <= 4.5
    ind_n = y >= 4.5
    x_outlier = np.array(x[ind_ol])
    y[ind_p] = 1.
    y[ind_n] = -1.
    x, y = np.array(x[ind]), y[ind]
    clf = svm.SVC(kernel="linear", cache_size=2000)
    clf.fit(x, y)
    y_outlier = clf.predict(x_outlier) * -1
    return x, y, x_outlier, y_outlier


def load_madelon():
    x1 = np.loadtxt("data/UCI/madelon/madelon_valid.data")
    x2 = np.loadtxt("data/UCI/madelon/madelon_train.data")
    y1 = np.loadtxt("data/UCI/madelon/madelon_valid.labels")
    y2 = np.loadtxt("data/UCI/madelon/madelon_train.labels")
    x = np.vstack([x1, x2])
    y = np.hstack([y1, y2])
    return x, y, None


def load_connect4():
    data = fetch_mldata("connect-4")
    x, y = data["data"].todense(), data["target"]
    x_outlier = x[y == 0]
    ind = (y != 0)
    x, y = x[ind], y[ind].astype(float)
    np.random.seed(0)
    ind = np.random.choice(range(len(x)), 10000)
    x, y = x[ind], y[ind]
    clf = svm.SVC(kernel="linear", cache_size=2000)
    clf.fit(x, y)
    y_outlier = clf.predict(x_outlier) * -1
    return np.array(x), y, np.array(x_outlier), y_outlier


def load_dna():
    data = fetch_mldata("dna")
    x, y = data["data"], data["target"]
    x = x.todense()
    x_outlier = x[y == 1]
    ind = (y != 1)
    x, y = x[ind], y[ind].astype(float)
    y[y == 2] = 1.
    y[y == 3] = -1.
    clf = svm.SVC(kernel="linear", cache_size=2000)
    clf.fit(x, y)
    y_outlier = clf.predict(x_outlier) * -1
    return np.array(x), np.array(y), np.array(x_outlier), y_outlier


def load_internet_ad():
    df = pd.read_csv("data/UCI/internet_ad/ad.data", header=None, skipinitialspace=True)
    df[1558][df[1558] == "ad."] = 1.
    df[1558][df[1558] == "nonad."] = -1.
    df = df.iloc[:, 4:]
    x = np.array(df.iloc[:, :(len(df.columns) - 1)], dtype=float)
    y = np.array(df.iloc[:, len(df.columns) - 1], dtype=float)
    return x, y, None


def load_w6a():
    x, y = load_svmlight_file("data/libsvm/w6a/w6a")
    x = x.toarray()
    return x, y, None


def load_aloi():
    x, y = load_svmlight_file("data/libsvm/aloi/aloi.scale.bz2")
    x = x.todense()
    x_outlier = x[y == 999]
    ind = (1 <= y) * (y <= 100)
    x, y = x[ind], y[ind]
    y[(1 <= y) * (y <= 50)] = 1.
    y[(50 <= y) * (y <= 100)] = -1.
    clf = svm.SVC(kernel="linear")
    clf.fit(x, y)
    y_outlier = clf.predict(x_outlier) * -1.
    return np.array(x), y, np.array(x_outlier), y_outlier


if __name__ == "__main__":
    x, y, x_outlier, y_outlier = load_usps()
    from mysvm import svmutil
    print "nu_max: {}".format(svmutil.calc_nu_max(y))

