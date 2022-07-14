import numpy as np
import random
import inspect

from .constant import CET_ADDI, CET_MULT, FUN_PROD, FUN_COST, OPT_DEFAULT, RTS_CRS, RTS_VRS, OPT_LOCAL
from .utils import tools


def index_tr(k, i_kfold):
    '''
    calculate the index of training set
    '''
    
    i_kfold_without_k = i_kfold[:k] + i_kfold[(k + 1):]
    flatlist = [item for elem in i_kfold_without_k for item in elem]

    return flatlist


def MSE(alpha, beta, x_test, y_test):

    yhat = np.zeros((len(x_test),))
    for i in range(len(x_test)):
        yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).min(axis=0)

    mse = np.mean((y_test -  yhat)**2)

    return mse


def split(x, y, ratio):
    
    '''
    function that splits data to training set and test set
    x is n*d matrix, y is n-array vector where n denotes sample size and 
    d is the number of variables.
    '''

    N = x.shape[0]

    # index of train and test sample
    train_ind = np.random.choice(N, int(ratio * N), replace=False)
    test_ind = np.setdiff1d(np.arange(N), train_ind)

    xtrain = x[train_ind,:]
    ytrain = y[train_ind]

    xtest = x[test_ind,:]
    ytest = y[test_ind]

    return xtrain, xtest, ytrain, ytest

# def model_tr(estimator):
#     if estimator == CSVR.CSVR:
#         return estimator(y_tr, x_tr, z, cet, fun, rts)

def cross_val_score(estimator, x, y, kfold=5, shuffle=False, 
                    z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS, epsilon=0.01, C=2):

    '''return MSE score
    '''

    N = x.shape[0]

    if shuffle:
        ind = random.sample(range(len(y)), k=len(y))
    else:
        ind = list(range(0, N))
    
    m = len(y) // kfold
    i_kfold = [ind[i:i+m] for i in range(0, len(ind), m)]
    if len(i_kfold) > kfold:
        i_kfold[-2:] = [i_kfold[-2]+i_kfold[-1]]

    error = []
    for k in range(kfold):
        
        # index of training set and test set
        ind_te = i_kfold[k]
        ind_tr = index_tr(k, i_kfold)

        x_tr = x[ind_tr, :]  
        y_tr = y[ind_tr]   

        x_te = x[ind_te, :]
        y_te = y[ind_te] 

        if estimator.__name__ == "CSVR.CSVR":
            model = estimator(y_tr, x_tr, z, cet, fun, rts, epsilon, C)
        else:
            model = estimator(y_tr, x_tr, z, cet, fun, rts)
        # model = estimator(y_tr, x_tr, z, cet, fun, rts, epsilon, C)
        model.optimize()
        alpha, beta = model.get_alpha(), model.get_beta()
        error.append( MSE(alpha, beta, x_te, y_te) )

    cv = np.mean(np.array(error), axis=0)

    return cv
