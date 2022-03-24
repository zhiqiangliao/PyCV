import numpy as np
import DGP


def split(X, y, ratio):
    
    '''
    function that splits data to training set and test set
    X is n*d matrix, y is n-array vector where n denotes sample size and 
    d is the number of variables.
    '''

    N = X.shape[0]
    train_ind = np.random.choice(N, int(ratio * N), replace=False)
    test_ind = np.setdiff1d(np.arange(N), train_ind)

    Xtrain = X[train_ind,:]
    ytrain = y[train_ind]

    Xtest = X[test_ind,:]
    ytest = y[test_ind]

    return Xtrain, ytrain, Xtest, ytest

n=20
d=2
sig = 2

x, y, y_true = DGP.inputs(n, d, sig)

a,b,c,d = split(x, y, 0.8)