
import sys

import theano

import theano.tensor as T

import network

import numpy as np

import time

import scipy.optimize as spo

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

    def __init__(self, layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size, X_train, y_train, N_train, \
        X_val, y_val, N_val, mean_y_train, std_y_train):

        layer_sizes[ 0 ] = layer_sizes[ 0 ] + 1
        self.batch_size = batch_size
        self.N_train = N_train
        self.X_train = X_train
        self.y_train = y_train

        self.N_val = N_val
        self.X_val = X_val
        self.y_val = y_val

        self.mean_y_train = mean_y_train
        self.std_y_train = std_y_train
        self.n_samples = n_samples

        # We create the network

        self.network = network.Network(layer_sizes, n_samples, v_prior, N_train)

        # index to a batch

        index = T.lscalar()  

        # We create the input and output variables. The input will be a minibatch replicated n_samples times

        self.x = T.matrix('x')
        self.y = T.matrix('y', dtype = 'float32')

        # The logarithm of the values for the likelihood factors
        
        ll_train = self.network.log_likelihood_values(self.x, self.y, 0.0, 1.0)
        ll_val = self.network.log_likelihood_values(self.x, self.y, mean_y_train, std_y_train)

        # The energy function for black-box alpha

        self.estimate_marginal_ll = -1.0 * N_train / (self.x.shape[ 0 ] * alpha) * \
            T.sum(LogSumExp(alpha * (T.sum(ll_train, 2) - self.network.log_f_hat()), 0)+ \
                T.log(1.0 / n_samples)) - self.network.log_normalizer_q() + \
            self.network.log_Z_prior()

        # We create a theano function for updating q
        
        self.process_minibatch = theano.function([ index ], self.estimate_marginal_ll, \
            updates = adam(self.estimate_marginal_ll, self.network.params, learning_rate), \
            givens = { self.x: self.X_train[ index * batch_size: (index + 1) * batch_size ],
            self.y: self.y_train[ index * batch_size: (index + 1) * batch_size ] })

        # We create a theano function for making predictions

        self.error_minibatch_train = theano.function([ index ],
            T.sum((T.mean(self.network.output(self.x), 0, keepdims = True)[ 0, :, : ] - self.y)**2) / layer_sizes[ -1 ],
            givens = { self.x: self.X_train[ index * batch_size: (index + 1) * batch_size ],
            self.y: self.y_train[ index * batch_size: (index + 1) * batch_size ] })

        self.error_minibatch_val = theano.function([ index ],
            T.sum((T.mean(self.network.output(self.x), 0, keepdims = True)[ 0, :, : ] * std_y_train + mean_y_train - self.y)**2) / layer_sizes[ -1 ],
            givens = { self.x: self.X_val[ index * batch_size: (index + 1) * batch_size ],
            self.y: self.y_val[ index * batch_size: (index + 1) * batch_size ] })

        self.ll_minibatch_val = theano.function([ index ], T.sum(LogSumExp(T.sum(ll_val, 2), 0) + T.log(1.0 / n_samples)), \
            givens = { self.x: self.X_val[ index * batch_size: (index + 1) * batch_size ],
            self.y: self.y_val[ index * batch_size: (index + 1) * batch_size ] })

        self.ll_minibatch_train = theano.function([ index ], T.sum(LogSumExp(T.sum(ll_train, 2), 0) + T.log(1.0 / n_samples)), \
            givens = { self.x: self.X_train[ index * batch_size: (index + 1) * batch_size ],
            self.y: self.y_train[ index * batch_size: (index + 1) * batch_size ] })

        self.target_lbfgs_grad = theano.function([ ], T.grad(self.estimate_marginal_ll, self.network.params),
            givens = { self.x: self.X_train, self.y: self.y_train })

        self.target_lbfgs_objective = theano.function([ ], self.estimate_marginal_ll, givens = { self.x: self.X_train, self.y: self.y_train })

        self.predict = theano.function([ self.x ], self.network.output(self.x) * std_y_train[ None, None, : ] + mean_y_train[ None, None, : ])

        self.network.update_randomness()

    def sample_predictive_distribution(self, x):

        noise_std = np.tile(np.expand_dims(self.std_y_train * \
            np.sqrt(np.exp(self.network.log_v_noise.get_value())), 0), [ self.n_samples, x.shape[ 0 ], 1 ])
        y = self.predict(x)
        return y + noise_std * np.random.randn(self.n_samples, x.shape[ 0 ], len(self.std_y_train))

    def predictive_distribution(self, x):

        for idxs in range(x.shape[0]):
            y = self.predict(idxs)
        return y 

    def train_LBFGS(self, n_epochs):

        initial_params = theano.function([ ], self.network.params)()

        params_shapes = [ s.shape for s in initial_params ]

        def de_vectorize_params(params):
            ret = []
            for shape in params_shapes:
                if len(shape) == 2 or len(shape) == 3:
                    ret.append(params[ : np.prod(shape) ].reshape(shape))
                    params = params[ np.prod(shape) : ]
                elif len(shape) == 1:
                    ret.append(params[ : np.prod(shape) ])
                    params = params[ np.prod(shape) : ]
                else:
                    ret.append(params[ 0 ])
                    params = params[ 1 : ]
            return ret

        def vectorize_params(params):
            return np.concatenate([ np.array(s).flatten() for s in params ])

        def set_params(params):
            for i in range(len(params)):
                self.network.params[ i ].set_value(params[ i ])

        def objective(params):
            params = np.array(params).astype(theano.config.floatX)
            params = de_vectorize_params(params)
            set_params(params)
            obj = np.array(self.target_lbfgs_objective(), dtype = np.float64)
            grad = np.array(vectorize_params(self.target_lbfgs_grad()), dtype = np.float64)
            return obj, grad

        initial_params = vectorize_params(initial_params)
        x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(objective, initial_params, bounds = None, iprint = 1, maxiter = n_epochs)
        set_params(de_vectorize_params(np.array(x_opt).astype(theano.config.floatX)))

        n_batches_train = np.int(np.ceil(1.0 * self.N_train / self.batch_size))
        n_batches_val = np.int(np.ceil(1.0 * self.N_val / self.batch_size))

        error_train = 0
        ll_train = 0
        for idxs in range(n_batches_train):
            error_train += self.error_minibatch_train(idxs)
            ll_train += self.ll_minibatch_train(idxs)
        error_train /= self.N_train
        error_train = np.sqrt(error_train)
        ll_train /= self.N_train

        error_val = 0
        ll_val = 0
        for idxs in range(n_batches_val):
            error_val += self.error_minibatch_val(idxs)
            ll_val += self.ll_minibatch_val(idxs)
        error_val /= self.N_val
        error_val = np.sqrt(error_val)
        ll_val /= self.N_val

        return error_val, ll_val

    def train_ADAM(self, n_epochs):

        n_batches_train = np.int(np.ceil(1.0 * self.N_train / self.batch_size))
        n_batches_val = np.int(np.ceil(1.0 * self.N_val / self.batch_size))
        for i in range(n_epochs):
            permutation = np.random.choice(range(n_batches_train), n_batches_train, replace = False)
            start = time.time()
            for idxs in range(n_batches_train):
#                if idxs % 10 == 9:
#                    self.network.update_randomness()
                self.network.update_randomness()
                ret = self.process_minibatch(permutation[ idxs ])
            end = time.time()

            # We evaluate the performance on the test data

            error_train = 0
            ll_train = 0
            for idxs in range(n_batches_train):
                error_train += self.error_minibatch_train(idxs)
                ll_train += self.ll_minibatch_train(idxs)
            error_train /= self.N_train
            error_train = np.sqrt(error_train)
            ll_train /= self.N_train

            error_val = 0
            ll_val = 0
            for idxs in range(n_batches_val):
                error_val += self.error_minibatch_val(idxs)
                ll_val += self.ll_minibatch_val(idxs)
            error_val /= self.N_val
            error_val = np.sqrt(error_val)
            ll_val /= self.N_val

            print(i, error_train, ll_train, error_val, ll_val, end - start)

        error_train = 0
        ll_train = 0
        for idxs in range(n_batches_train):
            error_train += self.error_minibatch_train(idxs)
            ll_train += self.ll_minibatch_train(idxs)
        error_train /= self.N_train
        error_train = np.sqrt(error_train)
        ll_train /= self.N_train

        error_val = 0
        ll_val = 0
        for idxs in range(n_batches_val):
            error_val += self.error_minibatch_val(idxs)
            ll_val += self.ll_minibatch_val(idxs)
        error_val /= self.N_val
        error_val = np.sqrt(error_val)
        ll_val /= self.N_val

        return error_val, ll_val
