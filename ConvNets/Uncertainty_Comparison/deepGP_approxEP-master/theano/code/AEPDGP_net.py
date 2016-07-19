import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt

import AEPDGP


class AEPDGP_net:

    def __init__(self, x_train, y_train, n_hiddens, n_inducing, normalise_x=False, normalise_y=True):

        # this is copied from PBP_net (Miguel's code)
        # We normalise the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalise_x:
            self.std_X_train = np.std(x_train, 0)
            self.std_X_train[self.std_X_train == 0] = 1
            self.mean_X_train = np.mean(x_train, 0)
        else:
            self.std_X_train = np.ones(x_train.shape[1])
            self.mean_X_train = np.zeros(x_train.shape[1])

        # x_train = (x_train - np.full(x_train.shape, self.mean_X_train)) / \
        #     np.full(x_train.shape, self.std_X_train)

        x_train = (x_train - self.mean_X_train) / self.std_X_train

        # x_train = x_train

        if normalise_y:
            self.mean_y_train = np.mean(y_train)
            self.std_y_train = np.std(y_train)
        else:
            self.mean_y_train = 0
            self.std_y_train = 1

        y_train_normalised = (y_train - self.mean_y_train) / self.std_y_train

        #y_train_normalised = y_train

        # we assume that we only deal with the single output regression case
        dims = np.concatenate(([x_train.shape[1]], n_hiddens, [1]))

        # construct the network
        Ntrain = x_train.shape[0]
        self.aepdgp_obj = AEPDGP.AEPDGP(Ntrain, dims.astype(int), n_inducing,
                                        self.mean_y_train, self.std_y_train)
        self.x_train = x_train
        self.y_train_normalised = y_train_normalised
        self.y_train = y_train

    def predict(self, x_test):

        """

            @param x_test   The matrix of features for the test data


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.

            Notes: from Miguel's PBP library
        """

        x_test = np.array(x_test, ndmin=2)

        # We normalise the test set

        # x_test = (x_test - np.full(x_test.shape, self.mean_X_train)) / \
        #     np.full(x_test.shape, self.std_X_train)

        x_test = (x_test - self.mean_X_train) / self.std_X_train

    
        # x_test = x_test

        # We compute the predictive mean and variance for the target variables
        # of the test data

        m, v = self.aepdgp_obj.predict(x_test)

        # We are done!

        return m, v

    def train(self, x_test=None, y_test=None,
              no_iterations=1000, no_points_per_mb=100, lrate=0.001):
        if x_test is None or y_test is None:
            compute_test = False
        else:
            compute_test = True

        # init hypers and inducing points
        self.aepdgp_obj.network.init_hypers(self.x_train, self.y_train_normalised)
        
        test_nll, test_rms, energy = \
            self.aepdgp_obj.train(self.x_train, self.y_train_normalised,
                                  x_test, y_test,
                                  no_iters=no_iterations,
                                  n_per_mb=no_points_per_mb,
                                  lrate=lrate, compute_test=compute_test)
        return test_nll, test_rms, energy


    def add_training_points(self, X_add, y_add):
        N_add = X_add.shape[0]
        for i in range(N_add):
            self.add_training_point(X_add[i, :], y_add[i])

    def add_training_point(self, X_add, y_add):
        # add new training point(s) to training set
        self.x_train = np.vstack((self.x_train, X_add))
        self.y_train = np.hstack((self.y_train, y_add))
        self.mean_y_train = np.mean(self.y_train)
        self.std_y_train = np.std(self.y_train)

        self.y_train_normalised = (self.y_train - self.mean_y_train) / self.std_y_train

        #self.y_train_normalised = self.y_train

        self.aepdgp_obj.std_y_train = self.std_y_train
        self.aepdgp_obj.mean_y_train = self.mean_y_train
        for layer in self.aepdgp_obj.network.layers:
            layer.Ntrain += X_add.shape[0]