# -*- coding: utf-8 -*-
"""
This file has contained the implementation of six methods.
"""
import numpy as np
from data_preprocess import *
from proj1_helpers import *

def compute_mse(y,tx,w):
    """Compute the mean square error."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient for least_square_GD and least_square_SGD."""
    n=y.size
    e=y-tx.dot(w)
    gradient=-1/n*(np.transpose(tx).dot(e))
    
    return gradient

def calculate_log_reg_gradient(y, tx, w):
    """compute the gradient of loss in the logistic regression."""

    return np.transpose(tx).dot(sigmoid(tx.dot(w)) - y)


def least_squares(y, tx):
    """use the least squares to find optimal value.
    param: y is the label value
           tx is the input data
    return: the optimal weights and the loss
    """
    w = np.linalg.solve(np.transpose(tx).dot(tx),np.transpose(tx).dot(y))
    loss = compute_mse(y,tx,w)
    
    return w,loss

def least_square_GD(y, tx, initial_w, max_iters, gamma):
    """Least square based on gradient descent algorithm.
    param: y is the label value
           tx is the input data
           initial_w is the initail value for w to iterate
           max_iters is the maximum number of iteration
           gamma is the learning rate
    return:the optimal weights and the loss
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):        
        gradient=compute_gradient(y, tx, w)
        loss= compute_mse(y, tx, w)
        w=w-gamma*gradient
        ws.append(w)
        losses.append(loss)

    return w, loss


def least_square_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Least square based on stochastic gradient descent algorithm.
    param: y is the label value
           tx is the input data
           initial_w is the initail valu for w to iterate
           batch_size is the subset of training sample
           max_iters is the maximum number of iteration
           gamma is the learning rate
    return:the optimal weights and the loss
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient=compute_gradient(minibatch_y, minibatch_tx, w)
            loss= compute_mse(minibatch_y, minibatch_tx, w)
            w=w-gamma*gradient
            ws.append(w)
            losses.append(loss)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """ ridge regression.
    param: y is the label value
           tx is the input data
           lambda_ is tradeoff parameter
    return:the optimal weights and the loss
    """
    w=np.linalg.solve((np.transpose(tx).dot(tx)+lambda_*2*tx.shape[0]*np.identity(tx.shape[1])),np.transpose(tx).dot(y))
    loss=compute_mse(y,tx,w)
    
    return w, loss




def sigmoid(x):
    """compute the sigmoid function
    """
    # clip the max input value so that exp(x) is not overflow
    x[x>25.0]=25.0
    x[x<-25.0]=-25.0
    
    return np.exp(x)/(np.exp(x)+1)

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    tx_dot_w = tx.dot(w)
    return np.sum(np.log(1. + np.exp(tx_dot_w)) - y * tx_dot_w)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_log_reg_gradient(y, tx, w)
    w -= gamma * gradient
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """logistic_regression
    param: y is the label value
           tx is the input data
           initial_w is the initail value for w to iterate
           max_iters is the maximum number of iteration
           gamma is the learning rate
    return:the optimal weights and the loss        
    """
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 200== 0:
            accu = accuracy(y, tx, w)
            print("Current iteration={i}, accuracy={l}".format(i=iter, l=accu))            
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + (lambda_ / 2) * np.transpose(w).dot(w)
    gradient =calculate_log_reg_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return w,loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """logistic_regression
    param: y is the label value
           tx is the input data
           lambda_ is the tradeoff parameter
           initial_w is the initail value for w to iterate
           max_iters is the maximum number of iteration
           gamma is the learning rate
    return:the optimal weights and the loss        
    """
    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w,loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if iter % 100 == 0:
            accu = accuracy(y, tx, w)
            print("Current iteration={i}, accuracy={l}".format(i=iter, l=accu))   
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, loss
