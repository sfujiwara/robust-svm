import matplotlib.pyplot as plt
import time
import pandas as pd

from src import ersvmdca, rampsvm, enusvm, ersvmutil

if __name__ == '__main__':
    filename = 'datasets/LIBSVM/cod-rna/cod-rna.csv'
    ## filename = 'datasets/LIBSVM/cod-rna/cod-rna_scale.csv'
    dataset = np.loadtxt(filename, delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    ersvmutil.libsvm_scale(x)
    ## Scaling
    ## for i in xrange(len(x[0])):
    ##     x[:,i] -= np.mean(x[:,i])
    ##     x[:,i] /= np.std(x[:,i])

    np.random.seed(1)
    ind_train = np.random.choice(len(y), 900, replace=False)
    num, dim = x.shape
    x_train = x[ind_train]
    y_train = y[ind_train]

    initial_weight = np.random.normal(size=dim)
    initial_weight = initial_weight / np.linalg.norm(initial_weight)

    print 'Ramp Loss SVM'
    ## ramp = rampsvm.RampSVM()
    ## ramp.solve_rampsvm(x_train, y_train)
    ## ramp.show_result()

    print 'ER-SVM'
    ersvm = ersvmdca.LinearPrimalERSVM()
    ersvm.set_initial_point(initial_weight, 0)
    ersvm.solve_ersvm(x, y)
    ersvm.show_result()

    print 'Enu-SVM'
    ## enu = enusvm.EnuSVM()
    ## enu.set_initial_weight(initial_weight)
    ## enu.solve_enusvm(x, y)
    ## print enu.comp_time

    print 'LIBSVM'
    start = time.time()
    libsvm = svm.SVC(C=1.)
    libsvm.fit(x, y)
    end = time.time()
    print 'end liblinear'
    print 'time:', end - start
