import numpy as np
import random
import DGP


# calculate the index of training set
def index_tr(k, i_kfold):
    
    i_kfold_without_k = i_kfold[:k] + i_kfold[(k + 1):]
    flatlist = [item for elem in i_kfold_without_k for item in elem]

    return flatlist

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
 
def kfold(x, y, fold, shuffle=False):

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

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for k in range(fold):
        
        # index of training set and test set
        ind_te = i_kfold[k]
        ind_tr = index_tr(k, i_kfold)

        # training predictors, training responses
        x_tr = x[ind_tr, :]  
        y_tr = y[ind_tr]   
        # validation predictors, validation responses
        x_te = x[ind_te, :]
        y_te = y[ind_te] 

    x_train.append(x_tr)
    x_test.append(x_te)
    y_train.append(y_tr)
    y_test.append(y_te)

    return x_train, x_test, y_train, y_test





n=26
d=2
sig = 2

x, y, y_true = DGP.inputs(n, d, sig)

# a,b,c,d = split(x, y, 0.8)
fold = 5
a,b,c,d = kfold(x, y, fold, shuffle=False)