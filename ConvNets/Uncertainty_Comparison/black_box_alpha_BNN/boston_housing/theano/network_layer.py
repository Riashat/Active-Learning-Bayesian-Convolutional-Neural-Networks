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
       
        scale = 0.1
        self.mean_param_W = theano.shared(value = scale * np.random.randn(1, d_in, d_out).astype(theano.config.floatX), name='m_W_par', borrow = True)
        self.log_var_param_W = theano.shared(value = -10 * np.ones((1, d_in, d_out)).astype(theano.config.floatX), name='log_v_W_par', borrow = True)
        scale = 0.1
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
            o = a

        return o

    @staticmethod
    def n_pdf(x):

        return 1.0 / T.sqrt(2 * math.pi) * T.exp(-0.5 * x**2)

    @staticmethod
    def n_cdf(x):

        return 0.5 * (1.0 + T.erf(x / T.sqrt(2.0)))

    @staticmethod
    def gamma(x):

        return Network_layer.n_pdf(x) / Network_layer.n_cdf(-x)

    @staticmethod
    def beta(x):

        return Network_layer.gamma(x) * (Network_layer.gamma(x) - x)

    def output_probabilistic(self, m_x, v_x):

        m_linear = T.dot(m_x, self.m_W[ 0, :, : ]) + T.tile(self.m_b[ 0, :, : ], [ m_x.shape[ 0 ], 1 ])
        v_linear = T.dot(m_x**2, self.v_W[ 0, :, : ]) + T.dot(v_x, self.m_W[ 0, :, : ]**2) + T.dot(v_x, self.v_W[ 0, :, : ]) + \
            T.tile(self.v_b[ 0, :, : ], [ m_x.shape[ 0 ], 1 ])

        if not self.output_layer:

            # We compute the mean and variance after the ReLU activation

            alpha = m_linear / T.sqrt(v_linear)
            gamma = Network_layer.gamma(-alpha)
            gamma_robust = -alpha - 1.0 / alpha + 2.0 / alpha**3
            gamma_final = T.switch(T.lt(-alpha, T.fill(alpha, 30)), gamma, gamma_robust)

            v_aux = m_linear + T.sqrt(v_linear) * gamma_final

            m_a = Network_layer.n_cdf(alpha) * v_aux
            v_a = m_a * v_aux * Network_layer.n_cdf(-alpha) + Network_layer.n_cdf(alpha) * v_linear * (1 - gamma_final * (gamma_final + alpha))

            return (m_a, v_a)

        else:

            return (m_linear, v_linear)

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
