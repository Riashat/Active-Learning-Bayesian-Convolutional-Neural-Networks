import numpy as np

import theano

import math

import theano.tensor as T

from network_layer import Network_layer

class Network:

    def __init__(self, layer_sizes, n_samples, v_prior, N):

        self.n_samples = n_samples
        self.layer_sizes = layer_sizes
        self.N = N
        self.v_prior = v_prior
        self.log_v_noise = theano.shared(value = np.zeros((1, layer_sizes[ -1 ])).astype(theano.config.floatX), name = 'log_v_noise', borrow = True)
        self.randomness_z = theano.shared(value = np.zeros((n_samples, N, 1)).astype(theano.config.floatX), name = 'z', borrow = True)

        # We create the different layers

        self.layers = []
        for d_in, d_out, layer_type in zip(layer_sizes[ : -1 ], layer_sizes[ 1 : ], [ False ] * (len(layer_sizes) - 2) + [ True ]):
            self.layers.append(Network_layer(d_in, d_out, n_samples, v_prior, N, layer_type))

        # We create the mean and variance parameters for all layers

        self.params = []
        for layer in self.layers:
            self.params.append(layer.mean_param_W)
            self.params.append(layer.log_var_param_W)
            self.params.append(layer.mean_param_b)
            self.params.append(layer.log_var_param_b)
        self.params.append(self.log_v_noise)

    def update_randomness(self):

        self.randomness_z.set_value(np.sqrt(self.layer_sizes[ 0 ] - 1) * np.float32(np.random.randn(self.n_samples, self.N, 1)))

        for layer in self.layers:
            layer.update_randomness()

    def output(self, x):
        
        x = T.tile(T.shape_padaxis(x, 0), [ self.n_samples, 1, 1 ])
        x = T.concatenate((x, 0 * self.randomness_z[ : , 0 : x.shape[ 1 ], : ]), 2)

        for layer in self.layers:
            x = layer.output(x)

        return x

    def log_likelihood_values(self, x, y, location = 0.0, scale = 1.0):

        o = self.output(x)
        noise_variance = T.tile(T.shape_padaxis(T.exp(self.log_v_noise[ 0, : ]) * scale**2, 0), [ o.shape[ 0 ], o.shape[ 1 ], 1])
        location = T.tile(T.shape_padaxis(location, 0), [ o.shape[ 0 ], o.shape[ 1 ], 1])
        scale = T.tile(T.shape_padaxis(scale, 0), [ o.shape[ 0 ], o.shape[ 1 ], 1])
        return -0.5 * T.log(2 * math.pi * noise_variance) - \
            0.5 * (o * scale + location - T.tile(T.shape_padaxis(y, 0), [ o.shape[ 0 ], 1, 1 ]))**2 / noise_variance

    def log_normalizer_q(self):

        ret = 0
        for layer in self.layers:
            ret += layer.log_normalizer_q()

        return ret

    def log_Z_prior(self):

        n_weights = 0
        for layer in self.layers:
            n_weights += layer.get_n_weights()

        return n_weights * (0.5 * np.log(self.v_prior * 2 * math.pi))

    def log_f_hat(self):

        ret = 0
        for layer in self.layers:
            ret += layer.log_f_hat()

        return ret
