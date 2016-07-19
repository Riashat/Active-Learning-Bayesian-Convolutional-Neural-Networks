
import theano
import theano.tensor as T
from black_box_alpha import BB_alpha

import os

import sys

import numpy as np

import gzip

import cPickle

# We download the data

def load_data():

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

        return shared_x, T.cast(shared_y, 'float32')

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
    y_train = y[ index_train.tolist() ][ : , None ]
    X_test = X[ index_test.tolist(), ]
    y_test = y[ index_test.tolist() ][ : , None ]

    # We normalize the features

    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    mean_X_train = np.mean(X_train, 0)
    X_train = (X_train - mean_X_train) / std_X_train
    X_test = (X_test - mean_X_train) / std_X_train
    mean_y_train = np.mean(y_train, 0)
    std_y_train = np.std(y_train, 0)

    y_train = (y_train - mean_y_train) / std_y_train

    train_set = X_train, y_train
    test_set = X_test, y_test

    train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval, train_set[ 0 ].shape[ 0 ], train_set[ 0 ].shape[ 1 ], mean_y_train, std_y_train

import os
if not os.path.isfile('results/test_error.txt') or not os.path.isfile('results/test_ll.txt'):

    os.system('rm results/*')

    # We load the random seed

    np.random.seed(1)

    # We load the data

    datasets, n, d, mean_y_train, std_y_train = load_data()

    train_set_x, train_set_y = datasets[ 0 ]
    test_set_x, test_set_y = datasets[ 1 ]

    N_train = train_set_x.get_value(borrow = True).shape[ 0 ]
    N_test = test_set_x.get_value(borrow = True).shape[ 0 ]
    layer_sizes = [ d, 100, len(mean_y_train) ]
    n_samples = 50
    alpha = 0.0001
    learning_rate = 0.001
    v_prior = 1.0
    batch_size = 32
    print '... building model'
    sys.stdout.flush()
    bb_alpha = BB_alpha(layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size, \
        train_set_x, train_set_y, N_train, test_set_x, test_set_y, N_test, mean_y_train, std_y_train)
    print '... training'
    sys.stdout.flush()

    test_error, test_ll = bb_alpha.train_ADAM(500)

    print('Test Error', test_error)
    print('Test Log Likelihood', test_ll)


    # with open("results/test_ll.txt", "a") as myfile:
    #     myfile.write(repr(test_ll) + '\n')

    # with open("results/test_error.txt", "a") as myfile:
    #     myfile.write(repr(test_error) + '\n')
