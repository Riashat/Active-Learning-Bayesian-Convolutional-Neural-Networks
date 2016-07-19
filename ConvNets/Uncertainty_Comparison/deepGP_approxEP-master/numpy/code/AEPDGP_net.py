import sys
# sys.path.append('~/synced/code/epsdgp/lib/tools/')
# print sys.path
# from system_tools import *

import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

import AEPDGP


class AEPDGP_net:

    def __init__(self, x_train, y_train, n_hiddens, n_inducing, lik='Gaussian',
                 normalise_x=False, normalise_y=True, n_samples=None, zu_tied=False):

        self.normalise_x = normalise_x
        if normalise_x:
            self.std_X_train = np.std(x_train, 0)
            self.std_X_train[self.std_X_train == 0] = 1
            self.mean_X_train = np.mean(x_train, 0)
        else:
            self.std_X_train = np.ones(x_train.shape[1])
            self.mean_X_train = np.zeros(x_train.shape[1])

        # x_train = (x_train - np.full(x_train.shape, self.mean_X_train)) / \
        #     np.full(x_train.shape, self.std_X_train)

        x_train = x_train
        self.lik = lik

        if lik == 'Gaussian':
            # single output, real-valued regression
            if normalise_y:
                self.mean_y_train = np.mean(y_train, axis=0)
                self.std_y_train = np.std(y_train, axis=0)
            else:
                self.mean_y_train = 0
                self.std_y_train = 1

            if np.ndim(self.std_y_train) == 0:
                if self.std_y_train <= 0:
                    self.std_y_train = 1
            else:
                invalid = np.where(self.std_y_train <= 0)
                self.std_y_train[invalid[0]] = 1
            
            # y_train_normalised = (y_train - self.mean_y_train) / self.std_y_train
            # self.y_train_normalised = y_train_normalised

            y_train_normalised = y_train
            self.y_train_normalised = y_train_normalised


            dims = np.concatenate(([x_train.shape[1]], n_hiddens, [y_train.shape[1]]))
            self.n_classes = None

        elif lik == 'Probit':
            # binary classificaton with probit likelihood
            dims = np.concatenate(([x_train.shape[1]], n_hiddens, [1]))
            self.mean_y_train = None
            self.std_y_train = None
            self.y_train_normalised = y_train
            self.n_classes = None
        elif lik == 'Softmax':
            # multiclass classificaton with softmax likelihood
            dims = np.concatenate(([x_train.shape[1]], n_hiddens, [y_train.shape[1]]))
            self.mean_y_train = None
            self.std_y_train = None
            self.y_train_normalised = y_train
            self.n_classes = int(y_train.shape[1])

        # construct the network
        Ntrain = x_train.shape[0]
        self.sepdgp_obj = AEPDGP.AEPDGP(Ntrain, dims.astype(int), n_inducing,
                                        mean_y_train=self.mean_y_train,
                                        std_y_train=self.std_y_train,
                                        lik=lik, n_samples=n_samples,
                                        n_classes=self.n_classes,
                                        zu_tied=zu_tied)
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):

        x_test = np.array(x_test, ndmin=2)

        if self.normalise_x:
            # We normalise the test set
            x_test = (x_test - np.full(x_test.shape, self.mean_X_train)) / \
                np.full(x_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        m, v = self.sepdgp_obj.predict(x_test)

        # We are done!

        return m, v

    def predict_intermediate(self, x_test, layer_no):

        x_test = np.array(x_test, ndmin=2)

        if self.normalise_x:
            # We normalise the test set
            x_test = (x_test - np.full(x_test.shape, self.mean_X_train)) / \
                np.full(x_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data at an intermediate layer

        m, v = self.sepdgp_obj.predict_intermediate(x_test, layer_no)

        # We are done!

        return m, v

    def predict_given_inputs(self, x, layer_no):
        x = np.array(x, ndmin=2)
        m, v = self.sepdgp_obj.predict_given_inputs(x, layer_no)
        return m, v

    def train(self, x_test=None, y_test=None,
              no_epochs=500, no_points_per_mb=100, lrate=0.001,
              reinit_hypers=True, compute_test=False, compute_logZ=False):

        

        if x_test is None or y_test is None:
            x_test = self.x_train
            y_test = self.y_train

        if self.normalise_x:
            # We normalise the test set
            x_test = (x_test - np.full(x_test.shape, self.mean_X_train)) / \
                np.full(x_test.shape, self.std_X_train)

        # init hypers and inducing points
        if self.lik == 'Gaussian':
            self.sepdgp_obj.network.init_hypers_Gaussian(self.x_train, self.y_train_normalised)
        elif self.lik == 'Probit':
            self.sepdgp_obj.network.init_hypers_Probit(self.x_train)
        elif self.lik == 'Softmax':
            self.sepdgp_obj.network.init_hypers_Softmax(self.x_train, self.y_train)

        
        test_nll, test_rms, energy = \
            self.sepdgp_obj.train(self.x_train, self.y_train_normalised,
                                  x_test, y_test,
                                  no_epochs=no_epochs,
                                  n_per_mb=no_points_per_mb,
                                  lrate=lrate,
                                  reinit_hypers=reinit_hypers,
                                  compute_test=compute_test,
                                  compute_logZ=compute_logZ)

        return test_nll, test_rms, energy



    def add_training_points(self, X_add, y_add):
        N_add = X_add.shape[0]
        for i in range(N_add):
            self.add_training_point(X_add[i, :], y_add[i, :])

    def add_training_point(self, X_add, y_add):
        y_add = np.reshape(y_add, (1, y_add.shape[0]))
        # add new training point(s) to training set
        self.x_train = np.vstack((self.x_train, X_add))
        self.y_train = np.vstack((self.y_train, y_add))
        self.mean_y_train = np.mean(self.y_train, axis=0)
        self.std_y_train = np.std(self.y_train, axis=0)
        #self.y_train_normalised = (self.y_train - self.mean_y_train) / self.std_y_train

        self.y_train_normalised = y_train

        self.sepdgp_obj.std_y_train = self.std_y_train
        self.sepdgp_obj.mean_y_train = self.mean_y_train
        self.sepdgp_obj.network.Ntrain += 1
