# -*- coding: utf-8 -*-
"""
This file is the final script file to get an exact submission.
"""
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from data_preprocess import *
from implementations import ridge_regression

print('Ridge regression is running... Please wait \n')

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


def ridge_regression_with_feature_augmentation(y_train, tx_train, degree):
    # generate a polynomial basis for train data to extend the features
    tx_train = build_poly(tx_train, degree)

    # do the riidge regression and compute the accuracy of the model
    w, loss = ridge_regression(y_train, tx_train, 0.004)
    accu_train = accuracy(y_train, tx_train, w)
    print('The train accuracy of ridge regression model is {accuracy}'.format(accuracy=accu_train))
    
    return w
                       
                       
w = ridge_regression_with_feature_augmentation(y, standardized_tx_train,12)

#  generate a polynomial basis for test data to extend the features
standardized_tx_test = build_poly(tx_test,15)

# generate predictions of testset
y_pred = predict_labels(w, standardized_tx_test)

# generate the submission file and save it
OUTPUT_PATH = 'ridge_regression_output.csv' 
create_csv_submission(ids_te, y_pred, OUTPUT_PATH)

print('Done')



