import numpy as np
import random

import DGP
import CR


# calculate the index of training set
def index_tr(k, i_kfold):
    
    i_kfold_without_k = i_kfold[:k] + i_kfold[(k + 1):]
    flatlist = [item for elem in i_kfold_without_k for item in elem]

    return flatlist


def MSE(alpha, beta, x_test, y_test):

    yhat = np.zeros((len(x_test),))
    for i in range(len(x_test)):
        yhat[i] = (alpha + np.matmul(beta, x_test[i].T)).min(axis=0)

    # y_est = alpha + np.sum(beta * x_test, axis=1)
    mse = np.mean((y_test -  yhat)**2)

    return mse


def split(x, y, ratio):
    
    '''
    function that splits data to training set and test set
    x is n*d matrix, y is n-array vector where n denotes sample size and 
    d is the number of variables.
    '''

    # the number of samples
    N = x.shape[0]

    # index of train and test sample
    train_ind = np.random.choice(N, int(ratio * N), replace=False)
    test_ind = np.setdiff1d(np.arange(N), train_ind)

    xtrain = x[train_ind,:]
    ytrain = y[train_ind]

    xtest = x[test_ind,:]
    ytest = y[test_ind]

    return xtrain, xtest, ytrain, ytest


def cross_val_score(estimator, x, y, kfold, shuffle=False):

    '''
    generate k-fold training and test data
    '''

    N = x.shape[0]

    if shuffle:
        ind = random.sample(range(len(y)), k=len(y))
    else:
        ind = list(range(0, N))
    
    m = len(y) // fold
    i_kfold = [ind[i:i+m] for i in range(0, len(ind), m)]
    if len(i_kfold) > kfold:
        i_kfold[-2:] = [i_kfold[-2]+i_kfold[-1]]

    error = []
    for k in range(fold):
        
        # index of training set and test set
        ind_te = i_kfold[k]
        ind_tr = index_tr(k, i_kfold)

        # train predictors, train responses
        x_tr = x[ind_tr, :]  
        y_tr = y[ind_tr]   

        # test predictors, test responses
        x_te = x[ind_te, :]
        y_te = y[ind_te] 

        model = estimator(y_tr, x_tr)
        model.optimize()
        alpha, beta = model.get_alpha(), model.get_beta()
        error.append( MSE(alpha, beta, x_te, y_te) )

    cv = np.mean(np.array(error), axis=0)

    return cv





n=50
d=2
sig = 2

x, y, y_true = DGP.inputs(n, d, sig)

# a,b,c,d = split(x, y, 0.8)
fold = 5
score = cross_val_score(CR.CR, x, y, fold, shuffle=False)
print(score)