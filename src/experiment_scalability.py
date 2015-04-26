import numpy as np
import ersvmdca

if __name__ == '__main__':
    ## Read a UCI dataset
    dataset = np.loadtxt('liver-disorders_scale.csv', delimiter=',')
    y = dataset[:, 0]
    x = dataset[:, 1:]
    num, dim = x.shape
    svm = ersvmdca.LinearPrimalERSVM()
    svm.set_initial_point(np.ones(dim), 0)
    svm.solve_ersvm(x, y)
    svm.show_result()
    
    
