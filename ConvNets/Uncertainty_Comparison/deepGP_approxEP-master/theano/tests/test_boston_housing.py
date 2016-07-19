import math
import numpy as np
import sys
import theano
sys.path.append('../code/')
import AEPDGP_net
import matplotlib.pyplot as plt
import time
import random

print 'Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir
# theano.config.compute_test_value = 'warn'
theano.config.optimizer = 'None'
# theano.exception_verbosity = 'high'
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
np.random.seed(1234)


# def step(x):
#     y = x.copy()
#     y[y < 0.0] = 0.0
#     y[y > 0.0] = 1.0
#     return y + 0.05*np.random.randn(x.shape[0], 1)

# number of GP layers
nolayers = 2
# number of hidden dimension in intermediate hidden layers
n_hiddens = [2]
# number of inducing points per layer
M = 20
n_pseudos = [M for _ in range(nolayers)]

notrain = 500
notest = 200
#no_iterations = 1000

no_iterations = 10

no_points_per_mb = 50

# X_train = np.reshape(np.linspace(-1, 1, notrain), (notrain, 1))
# X_test = np.reshape(np.linspace(-1, 1, notest), (notest, 1))
# X_plot = np.reshape(np.linspace(-1.5, 1.5, notest), (notest, 1))
# y_train = step(X_train)
# y_test = step(X_test)
# y_train = np.reshape(y_train, (notrain, ))
# y_test = np.reshape(y_test, (notest, ))


data = np.loadtxt('boston_housing.txt')

# We obtain the features and the targets

X = data[ :, range(data.shape[ 1 ] - 1) ]
y = data[ :, data.shape[ 1 ] - 1 ]

permutation = np.asarray(random.sample(range(0,X.shape[0]), X.shape[0]))

size_train = np.round(X.shape[ 0 ] * 0.9)
index_train = permutation[ 0 : size_train ]
index_test = permutation[ size_train : ]

X_train = X[ index_train, : ]
y_train = y[ index_train ]
X_test = X[ index_test, : ]
y_test = y[ index_test ]


# We construct the network
net = AEPDGP_net.AEPDGP_net(X_train, y_train, n_hiddens, n_pseudos)
t0 = time.time()
# train
test_nll, test_rms, energy = net.train(no_iterations=no_iterations,
                               no_points_per_mb=no_points_per_mb,
                               lrate=0.02)
t1 = time.time()
print 'time: ', t1 - t0

# We make predictions for the test set
m, v = net.predict(X_test)

# We compute the test RMSE
rmse = np.sqrt(np.mean((y_test - m)**2))
print 'test rmse: ', rmse

# We compute the test log-likelihood
test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v)) - \
    0.5 * (y_test - m)**2 / (v))
print 'test log-likelihood: ', test_ll

# m, v = net.predict(X_plot)
# plt.figure()
# plt.plot(X_train, y_train, 'bo', alpha=0.5)
# plt.plot(X_plot, m, 'm-')
# plt.plot(X_plot, m-2*np.sqrt(v), 'm+')
# plt.plot(X_plot, m+2*np.sqrt(v), 'm+')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.show()