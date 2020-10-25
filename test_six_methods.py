# -*- coding: utf-8 -*-
"""
This script is used to test the basis six implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from data_preprocess import *
from implementations import *

# test least square
def test_least_squares(y_train, tx_train,y_test,x_test):
    print('\Testing least_squares...')
    tx_train = build_poly(tx_train, 11)
    x_test = build_poly(x_test, 11)
    w, loss = least_squares(y_train, tx_train)
    accu_train=accuracy(y_train, tx_train, w)
    accu_test=accuracy(y_test, x_test, w)
    print('The train accuracy of least square model is {accuracy}'.format(accuracy=accu_train))
    print('The test accuracy of least square model is {accuracy}'.format(accuracy=accu_test))
    return w, loss

# test least square using gradient descent
def test_least_squares_GD(y_train, tx_train,y_test, x_test,max_iters, gamma):
    print('\Testing least_squares_GD...')

    initial_w=np.zeros(tx_train.shape[1])
    w, loss = least_square_GD(y_train, tx_train, initial_w, max_iters, gamma)
    accu_train=accuracy(y_train, tx_train, w)
    accu_test=accuracy(y_test, x_test, w)
    print('The train accuracy of least square using gradient descent model is {accuracy}'.format(accuracy=accu_train))
    print('The test accuracy of least square using gradient descent model is {accuracy}'.format(accuracy=accu_test))
    return w, loss

# test least square using stochastic gradient descent
def test_least_squares_SGD(y_train, tx_train,y_test, x_test,max_iters, gamma):
    print('\Testing least_squares_SGD...')
    batch_size = 1 
    initial_w=np.zeros(tx_train.shape[1])
    w, loss = least_square_SGD(y_train, tx_train, initial_w, batch_size, max_iters, gamma)
    accu_train=accuracy(y_train, tx_train, w)
    accu_test=accuracy(y_test, x_test, w)
    print('The train accuracy of least square using stochastic gradient descent model is {accuracy}'.format(accuracy=accu_train))
    print('The test accuracy of least square using stochastic gradient descent model is {accuracy}'.format(accuracy=accu_test))
    return w, loss

# test ridge regression
def test_ridge_regression(y_train, tx_train,y_test, x_test):
    print('\Testing ridge_regression...')
    w, loss=ridge_regression(y_train, tx_train, 0.01)
    accu_train=accuracy(y_train, tx_train, w)
    accu_test=accuracy(y_test, x_test, w)
    print('The train accuracy of ridge regression model is {accuracy}'.format(accuracy=accu_train))
    print('The test accuracy of ridge regression model is {accuracy}'.format(accuracy=accu_test))
    return w, loss

# test logistic regression
def test_logistic_regression(y_train, tx_train,y_test, x_test,max_iters, gamma):
    print('\Testing logistic_regression...')
    initial_w=np.zeros(tx_train.shape[1])
    w, loss = logistic_regression(y_train, tx_train, initial_w, max_iters, gamma)
    accu_train=accuracy(y_train, tx_train, w)
    accu_test=accuracy(y_test, x_test, w)
    print('The train accuracy of logistic_regression model is {accuracy}'.format(accuracy=accu_train))
    print('The test accuracy of logistic_regression model is {accuracy}'.format(accuracy=accu_test))
    return w, loss

# test regularized logistic regression
def test_reg_logistic_regression(y_train, tx_train,y_test, x_test,lambda_,max_iters, gamma):
    print('\Testing reg_least_squares_SGD...')
    initial_w=np.zeros(tx_train.shape[1])
    w, loss = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)
    accu_train=accuracy(y_train, tx_train, w)
    accu_test=accuracy(y_test, x_test, w)
    print('The train accuracy of regularized logistic_regression model is {accuracy}'.format(accuracy=accu_train))
    print('The test accuracy of regularized logistic_regression model is {accuracy}'.format(accuracy=accu_test))
    return w, loss
    

DATA_TRAIN_PATH = 'train.csv' 
DATA_TEST_PATH = 'test.csv' 

# Load the training data into feature matrix, class labels, and event ids:
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
y_test, tx_test, ids_te = load_csv_data(DATA_TEST_PATH)

# Replace -999 with median of that column and standardise the feature matrix
tX = replace_999_by_median(tX)
tx_test = replace_999_by_median(tx_test)
standardized_tx_train, mean_tx_train, std_tx_train = standardize(tX)
standardized_tx_test, mean_tx_test, std_tx_test = standardize(tx_test)

#split the training data int two parts. One is training set, which is used to train the model. The other is testing set, which can show if the model can fit the unknown data.
x_training, x_test,y_training, y_test = split_data(standardized_tx_train, y, 0.8, seed=5)


w1, loss1 = test_least_squares_GD(y_training, x_training,y_test, x_test, 3000, 0.01 )    
    
w2, loss2 = test_least_squares_SGD(y_training, x_training,y_test, x_test, 3000, 0.001 )

w3, loss3 = test_least_squares(y_training, x_training,y_test, x_test)

w4, loss4 = test_ridge_regression(y_training, x_training,y_test, x_test)

w5, loss5 = test_logistic_regression(y_training, x_training,y_test, x_test, 1000, 0.00000005)

w6, loss6 = test_reg_logistic_regression(y_training, x_training,y_test, x_test, 0.005, 1000, 0.00000005)




















