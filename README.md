# EPFL-ML-Project-1
## Kang Fu, Runke Zhou, Shanci Li 
This repository contains all the files used in the first project of [Maching Learning course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) at EPFL(2020 Fall).
This project is to deal with the actual CERN particle accelerator data, using Machine Learning algorithms to determine whether an event is a signal or background of Higgs boson.
Structure.

The project has been developed by Jupyter and tested with Python 3.6. In this project, the library  `matplotlib` for visualization and `NumPy` for matrix computation are mainly used. In order to reduplicate my result, please install the Python, Juputer and the relating library beforehand.

## Structure
- feature_distribution.ipynb: this jupyper notebook provides the feature distributions in the form of histogram. 
- project1_helpers.py: contains some helper functions for project 1 like functions to load the data and create the submission file.
- data_reprocess.py: contains the methods to preprocess the data, including remove outliers and standardization.
- cross_validation.py: cross validation for ridge regression with feature augmentation model
- implementations.py: contains all the function of implementations required by the project.
- test_six_methods.py: this script is used to test the implementation by calling it from implementations.py.
- test_implementation.ipynb: this jupyper notebook is used to test the implementation and tune the parameters.
- run.py: contains the selected model to train the model and can get the submission file.
- train.csv: the training data
- test.csv: the testing data
- README.md: this file :)



