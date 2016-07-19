import numpy as np

import theano

import math

import theano.tensor as T

from network_layer import Network_layer

class Network:

    def __init__(self, layer_sizes, n_samples, v_prior, N):

        self.n_samples = n_samples
        self.v_prior = v_prior

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

    def update_randomness(self):

        for layer in self.layers:
            layer.update_randomness()

    def output(self, x):

        x = T.tile(T.shape_padaxis(x, 0), [ self.n_samples, 1, 1 ])

        for layer in self.layers:
            x = layer.output(x)

        return x

    def log_likelihood_values(self, x, y):

        o = self.output(x)
        return o[ : , T.arange(y.shape[ 0 ]) , y ]

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
