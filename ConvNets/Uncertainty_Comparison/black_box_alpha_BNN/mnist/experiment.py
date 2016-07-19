
import theano
import theano.tensor as T
from black_box_alpha import BB_alpha

import os

import sys

import numpy as np

import gzip

import cPickle

# We download the data

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_set = (np.concatenate((train_set[ 0 ], valid_set[ 0 ]), 0), np.concatenate((train_set[ 1 ], valid_set[ 1 ]), 0))
    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    train_set_x, train_set_y = shared_dataset(train_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval, train_set[ 0 ].shape[ 0 ], train_set[ 0 ].shape[ 1 ], np.max(train_set[ 1 ]) + 1

import os
if not os.path.isfile('results/test_error.txt') or not os.path.isfile('results/test_ll.txt'):

    os.system('rm results/*')

    # We load the random seed

    seed = 1
    np.random.seed(seed)

    # We load the data

    datasets, n, d, n_labels = load_data('/tmp/mnist.pkl.gz')

    train_set_x, train_set_y = datasets[ 0 ]
    test_set_x, test_set_y = datasets[ 1 ]

    N_train = train_set_x.get_value(borrow = True).shape[ 0 ]
    N_test = test_set_x.get_value(borrow = True).shape[ 0 ]
    layer_sizes = [ d, 400, 400, n_labels ]
    n_samples = 50
    alpha = 0.5
    learning_rate = 0.0001
    v_prior = 1.0
    batch_size = 250
    print '... building model'
    sys.stdout.flush()
    bb_alpha = BB_alpha(layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size, \
        train_set_x, train_set_y, N_train, test_set_x, test_set_y, N_test)
    print '... training'
    sys.stdout.flush()

    test_error, test_ll = bb_alpha.train(250)

    with open("results/test_ll.txt", "a") as myfile:
        myfile.write(repr(test_ll) + '\n')

    with open("results/test_error.txt", "a") as myfile:
        myfile.write(repr(test_error) + '\n')
