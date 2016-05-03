# -*- coding: utf-8 -*-

from sklearn.datasets import load_svmlight_file


def load_data(name):
    if name == "mushrooms":
        return load_mushrooms()


def load_mushrooms():
    x, y = load_svmlight_file("data/LIBSVM/mushrooms/mushrooms")
    x = x.toarray()
    y[y > 1.5] = -1.
    return x, y