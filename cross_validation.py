# -*- coding: utf-8 -*-
"""
This is the cross validation for ridge regression with feature augmentation model.
"""

import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *


def split_data(x, y, ratio, seed=1):
    """split the dataset into two parts (train set and test set) based on the split ratio.
    param: x: input feature data
           y: input label data
           ratio: split ratio (equal to number of train set over number of total number)
           seed: random generation seed
    return: the feature and label data of train set and test set respectively
    """
    # set seed
    np.random.seed(seed)
    permutation = np.random.permutation(len(y))
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]
    split_position = int(len(y) * ratio)
    x_training, x_test = shuffled_x[ : split_position], shuffled_x[split_position : ]
    y_training, y_test = shuffled_y[ : split_position], shuffled_y[split_position : ]
    
    return x_training, x_test,y_training, y_test

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the accuracy of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    # ridge regression
    w, loss = ridge_regression(y_tr, tx_tr, lambda_)
    train_acc=accuracy(y_tr, tx_tr, w)
    test_acc=accuracy(y_te, tx_te, w)
    return train_acc, test_acc,w
            

def best_degree_selection(y, x, degrees, k_fold, lambdas, seed = 1):
    """Selection of the best degree """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    #for each degree, we compute the best lambdas and the associated accuracy
    best_lambdas = []
    best_accu = []
    #vary degree
    for degree in degrees:
        # cross validation
        accu_te = []
        for lambda_ in lambdas:
            accu_te_tmp = []
            for k in range(k_fold):
                train_acc, test_acc,w = cross_validation(y, x, k_indices, k, lambda_, degree)
                accu_te_tmp.append(test_acc)
            accu_te.append(np.mean(accu_te_tmp))
            print("test")
        ind_lambda_opt = np.argmax(accu_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_accu.append(accu_te[ind_lambda_opt])
        
    ind_best_degree =  np.argmax(best_accu)      
        
    return degrees[ind_best_degree]


def cross_validation_visualization(lambds, accu_tr, accu_te):
    """visualization the curves of train accuracy and test accuracy."""
    plt.semilogx(lambds, accu_tr, marker=".", color='b', label='train accu')
    plt.semilogx(lambds, accu_te, marker=".", color='r', label='test accu')
    plt.xlabel("lambda")
    plt.ylabel("Accuracy")
    plt.title("cross validation")
    plt.legend(["accu_tr","accu_te"],loc="upper right")
    plt.grid(True)
    plt.savefig("cross_validation")


            