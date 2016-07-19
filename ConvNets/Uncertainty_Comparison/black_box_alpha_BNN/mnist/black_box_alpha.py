
import sys

import theano

import theano.tensor as T

import network

import numpy as np

import time

def make_batches(N_data, batch_size):
    return [ slice(i, min(i + batch_size, N_data)) for i in range(0, N_data, batch_size) ]

def LogSumExp(x, axis = None):
    x_max = T.max(x, axis = axis, keepdims = True)
    return T.log(T.sum(T.exp(x - x_max), axis = axis, keepdims = True)) + x_max

def adam(loss, all_params, learning_rate = 0.001):
    b1 = 0.9
    b2 = 0.999
    e = 1e-8
    gamma = 1 - 1e-8
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        m = b1 * m_previous + (1 - b1) * g                           # (Update biased first moment estimate)
        v = b2 * v_previous + (1 - b2) * g**2                            # (Update biased second raw moment estimate)
        m_hat = m / (1 - b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1 - b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

class BB_alpha:

    def __init__(self, layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size, X_train, y_train, N_train, X_val, y_val, N_val):

        self.batch_size = batch_size
        self.N_train = N_train
        self.X_train = X_train
        self.y_train = y_train

        self.N_val = N_val
        self.X_val = X_val
        self.y_val = y_val

        # We create the network

        self.network = network.Network(layer_sizes, n_samples, v_prior, N_train)

        # index to a batch

        index = T.lscalar()  

        # We create the input and output variables. The input will be a minibatch replicated n_samples times

        self.x = T.matrix('x')
        self.y = T.vector('y', dtype = 'int32')

        # The logarithm of the values for the likelihood factors
        
        ll = self.network.log_likelihood_values(self.x, self.y)

        # The energy function for black-box alpha

        self.estimate_marginal_ll = -1.0 * N_train / (self.x.shape[ 0 ] * alpha) * \
            T.sum(LogSumExp(alpha * (ll - self.network.log_f_hat()), 0) + T.log(1.0 / n_samples)) - self.network.log_normalizer_q() + \
            self.network.log_Z_prior()

        # We create a theano function for updating q
        
        self.process_minibatch = theano.function([ index ], self.estimate_marginal_ll, \
            updates = adam(self.estimate_marginal_ll, self.network.params, learning_rate), \
            givens = { self.x: self.X_train[ index * batch_size: (index + 1) * batch_size ], \
            self.y: self.y_train[ index * batch_size: (index + 1) * batch_size ] })

        # We create a theano function for making predictions

        self.error_minibatch_train = theano.function([ index ],
            T.mean(T.neq(T.argmax((LogSumExp(self.network.output(self.x), 0) + T.log(1.0 / n_samples))[ 0, :, : ], axis = 1), self.y)),
            givens = { self.x: self.X_train[ index * batch_size: (index + 1) * batch_size ],
            self.y: self.y_train[ index * batch_size: (index + 1) * batch_size ] })

        self.error_minibatch_val = theano.function([ index ], 
            T.mean(T.neq(T.argmax((LogSumExp(self.network.output(self.x), 0) + T.log(1.0 / n_samples))[ 0, :, : ], axis = 1), self.y)),
            givens = { self.x: self.X_val[ index * batch_size: (index + 1) * batch_size ],
            self.y: self.y_val[ index * batch_size: (index + 1) * batch_size ] })

        self.ll_minibatch_val = theano.function([ index ], T.mean(LogSumExp(ll, 0) + T.log(1.0 / n_samples)), \
            givens = { self.x: self.X_val[ index * batch_size: (index + 1) * batch_size ], \
            self.y: self.y_val[ index * batch_size: (index + 1) * batch_size ] })

        self.network.update_randomness()

    def train(self, n_epochs):

        n_batches_train = self.N_train / self.batch_size
        n_batches_val = self.N_val / self.batch_size
        for i in range(n_epochs):
            permutation = np.random.choice(range(n_batches_train), n_batches_train, replace = False)
            for idxs in range(n_batches_train):
                start = time.time()
                if idxs % 10 == 9:
                    self.network.update_randomness()
                ret = self.process_minibatch(permutation[ idxs ])
                end = time.time()
                print i, idxs, end - start
                sys.stdout.flush()

            # We evaluate the performance on the test data

            error_val = 0
            ll_val = 0
            for idxs in range(n_batches_val):
                error_val += self.error_minibatch_val(idxs)
                ll_val += self.ll_minibatch_val(idxs)
            error_val /= n_batches_val
            ll_val /= n_batches_val

            error_train = 0
            for idxs in range(n_batches_train):
                error_train += self.error_minibatch_train(idxs)
            error_train /= n_batches_train

            print(i, error_train, error_val, ll_val)
            sys.stdout.flush()

        return error_val, ll_val
