
import math

import time

import numpy as np

import pickle
import gzip

from black_box_alpha import fit_q

# We fix the random seed

np.random.seed(1)

# We load the data

data = np.loadtxt('../data/data.txt')

# We load the indexes for the features and for the target

index_features = np.loadtxt('../data/index_features.txt')
index_target = np.loadtxt('../data/index_target.txt')

X = data[ : , index_features.tolist() ]
y = data[ : , index_target.tolist() ]

i = np.int(np.loadtxt('simulation_index.txt'))

# We load the indexes of the training set

index_train = np.loadtxt("../data/index_train_{}.txt".format(i))
index_test = np.loadtxt("../data/index_test_{}.txt".format(i))

X_train = X[ index_train.tolist(), ]
y_train = y[ index_train.tolist() ]
X_test = X[ index_test.tolist(), ]
y_test = y[ index_test.tolist() ]

# We normalize the features

std_X_train = np.std(X_train, 0)
std_X_train[ std_X_train == 0 ] = 1
mean_X_train = np.mean(X_train, 0)
X_train = (X_train - mean_X_train) / std_X_train
X_test = (X_test - mean_X_train) / std_X_train
mean_y_train = np.mean(y_train)
std_y_train = np.std(y_train)
y_train = (y_train - mean_y_train) / std_y_train

y_train = np.array(y_train, ndmin = 2).reshape((-1, 1))
y_test = np.array(y_test, ndmin = 2).reshape((-1, 1))

# We load the hypers

import os
if not os.path.isfile('results/test_error.txt') or not os.path.isfile('results/test_ll.txt'):

    learning_rate = 0.001
    v_prior = 1.0

    # We iterate the method 

    batch_size = 32
    epochs = 500
    K = 50

    start_time = time.time()
    w, v_prior, get_error_and_ll = fit_q(X_train, y_train, 100, batch_size, epochs, K, learning_rate, v_prior)
    running_time = time.time() - start_time

    # We obtain the test RMSE and the test ll

    error, ll = get_error_and_ll(w, v_prior, X_test, y_test, K, mean_y_train, std_y_train)

    with open("results/test_error.txt", "a") as myfile:
        myfile.write(repr(error) + '\n')

    with open("results/test_ll.txt", "a") as myfile:
        myfile.write(repr(ll) + '\n')
