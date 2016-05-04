# -*- coding: utf-8 -*-

from sklearn.datasets import load_svmlight_file


def load_data(name):
    if name == "mushrooms":
        return load_mushrooms()
    if name == "gisette":
        return load_gisette()
    if name == "usps":
        return load_usps()


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


if __name__ == "__main__":
    x, y, x_outlier = load_usps()
