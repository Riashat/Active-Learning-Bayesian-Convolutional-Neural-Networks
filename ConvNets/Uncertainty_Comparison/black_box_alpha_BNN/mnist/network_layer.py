import theano

import theano.tensor as T

import math

import numpy as np

class Network_layer:

    def __init__(self, d_in, d_out, n_samples, v_prior, N, output_layer = False):

        self.d_in = d_in
        self.d_out = d_out
        self.n_samples = n_samples
        self.output_layer = output_layer
        self.v_prior = v_prior
        self.N = N
       
#        scale = 0.1
        scale = np.sqrt(2.0 / (d_in + d_out))
        self.mean_param_W = theano.shared(value = scale * np.random.randn(1, d_in, d_out).astype(theano.config.floatX), name='m_W_par', borrow = True)
        self.log_var_param_W = theano.shared(value = -10 * np.ones((1, d_in, d_out)).astype(theano.config.floatX), name='log_v_W_par', borrow = True)
#        scale = 0.1
        scale = np.sqrt(2.0 / (1 + d_out))
        self.mean_param_b = theano.shared(value = scale * np.random.randn(1, 1, d_out).astype(theano.config.floatX), name='m_b_par', borrow = True)
        self.log_var_param_b = theano.shared(value = -10 * np.ones((1, 1, d_out)).astype(theano.config.floatX), name='log_v_b_par', borrow = True)

        self.randomness_W = theano.shared(value = np.zeros((n_samples, d_in, d_out)).astype(theano.config.floatX), name='e_W', borrow = True)
        self.randomness_b = theano.shared(value = np.zeros((n_samples, 1, d_out)).astype(theano.config.floatX), name='e_b', borrow = True)
        self.W = theano.shared(value = np.zeros((n_samples, d_in, d_out)).astype(theano.config.floatX), name='W', borrow = True)
        self.b = theano.shared(value = np.zeros((n_samples, 1, d_out)).astype(theano.config.floatX), name='b', borrow = True)

        # We create the variables with the means and variances

        self.m_W = theano.shared(value = np.zeros((1, d_in, d_out)).astype(theano.config.floatX), name='m_W', borrow = True)
        self.v_W = theano.shared(value = np.zeros((1, d_in, d_out)).astype(theano.config.floatX), name='v_W', borrow = True)
        self.m_b = theano.shared(value = np.zeros((1, 1, d_out)).astype(theano.config.floatX), name='m_b', borrow = True)
        self.v_b = theano.shared(value = np.zeros((1, 1, d_out)).astype(theano.config.floatX), name='v_b', borrow = True)

    def update_randomness(self):

        random_samples = np.random.randn(self.n_samples, self.d_in + 1, self.d_out)
        self.randomness_W.set_value(np.float32(random_samples[ :, : - 1, : ]))
        self.randomness_b.set_value(np.float32(random_samples[ :, -1 :, : ]))

    def logistic(self, x):
        return 1.0 / (1.0 + T.exp(-x))

    def update_sample_weights(self):

        # We update the mean and variances of q

        self.v_W = self.v_prior * self.logistic(self.log_var_param_W)
        self.m_W = self.mean_param_W

        self.v_b = self.v_prior * self.logistic(self.log_var_param_b)
        self.m_b = self.mean_param_b

        # We update the random samples for the network weights

        self.W = self.randomness_W * T.tile(T.sqrt(self.v_W), [ self.n_samples, 1, 1 ]) + T.tile(self.m_W, [ self.n_samples, 1, 1 ])
        self.b = self.randomness_b * T.tile(T.sqrt(self.v_b), [ self.n_samples, 1, 1 ]) + T.tile(self.m_b, [ self.n_samples, 1, 1 ])

    def output(self, x):

        self.update_sample_weights()

        # We compute the activations

        a = T.batched_dot(x, self.W) + T.tile(self.b, [ 1, x.shape[ 1 ], 1 ])

        if not self.output_layer:
            o = T.nnet.relu(a)
        else:
            a = a - a.max(axis = 2, keepdims = True)
            o = a - T.log(T.sum(T.exp(a), axis = 2, keepdims = True))

        return o

    def log_normalizer_q(self):
        
        logZ_W = T.sum(0.5 * T.log(self.v_W * 2 * math.pi) + 0.5 * self.m_W**2 / self.v_W)
        logZ_b = T.sum(0.5 * T.log(self.v_b * 2 * math.pi) + 0.5 * self.m_b**2 / self.v_b)
        return logZ_W + logZ_b

    def get_n_weights(self):
        return self.d_in * self.d_out + self.d_out

    def log_f_hat(self):

        v_W = 1.0 / (1.0 / self.N * (1.0 / self.v_W - 1.0 / self.v_prior))
        m_W = 1.0 / self.N * self.m_W / self.v_W * v_W
        v_b = 1.0 / (1.0 / self.N * (1.0 / self.v_b - 1.0 / self.v_prior))
        m_b = 1.0 / self.N * self.m_b / self.v_b * v_b

        log_f_hat_W = T.sum(-0.5 * T.tile(1.0 / v_W, [ self.n_samples, 1, 1 ]) * self.W**2 + \
            T.tile(m_W / v_W, [ self.n_samples, 1, 1 ]) * self.W, axis = [ 1, 2 ], keepdims = True)[ :, :, 0 ]
        log_f_hat_b = T.sum(-0.5 * T.tile(1.0 / v_b, [ self.n_samples, 1, 1 ]) * self.b**2 + \
            T.tile(m_b /  v_b, [ self.n_samples, 1, 1 ]) * self.b, axis = [ 1, 2 ], keepdims = True)[ :, :, 0 ]

        return log_f_hat_W + log_f_hat_b
