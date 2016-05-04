# -*- coding: utf-8 -*-

from sklearn.datasets import load_svmlight_file
from sklearn.datasets.mldata import fetch_mldata
import numpy as np


def load_data(name):
    if name == "mushrooms":
        return load_mushrooms()
    if name == "gisette":
        return load_gisette()
    if name == "usps":
        return load_usps()
    if name == "dna":
        return load_dna()


def load_mushrooms():
    x, y = load_svmlight_file("data/libsvm/mushrooms/mushrooms")
    x = x.toarray()
    y[y > 1.5] = -1.
    return x, y


def load_gisette():
    x, y = load_svmlight_file("data/libsvm/gisette/gisette_scale.bz2")
    x = x.toarray()
    return x, y


def load_usps():
    x, y = load_svmlight_file("data/libsvm/usps/usps.bz2")
    x = x.toarray()
    x_outlier = x[y == 2]
    ind = (y != 2)
    x, y = x[ind], y[ind]
    y[y != 1] = -1.
    return x, y, x_outlier


def load_connect4():
    data = fetch_mldata("connect-4")
    x, y = data["data"], data["target"]
    x_outlier = x[y == 0]
    ind = (y != 0)
    x, y = x[ind].todense(), y[ind].astype(float)
    return x, y, x_outlier


def load_dna():
    data = fetch_mldata("dna")
    x, y = data["data"], data["target"]
    x = x.todense()
    x_outlier = x[y == 1]
    ind = (y != 1)
    x, y = x[ind], y[ind].astype(float)
    y[y == 2] = 1.
    y[y == 3] = -1.
    return np.array(x), np.array(y), np.array(x_outlier)


if __name__ == "__main__":
    x, y, x_outlier = load_dna()
