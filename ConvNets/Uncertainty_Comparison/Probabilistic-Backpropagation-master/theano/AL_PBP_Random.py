
import math

import numpy as np

import sys
sys.path.append('PBP_net/')
import PBP_net

import random


import timeit


import theano


start = timeit.default_timer()


# print 'Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir
# # theano.config.compute_test_value = 'warn'
# theano.config.optimizer = 'None'
# # theano.exception_verbosity = 'high'
# theano.config.mode = 'FAST_RUN'
# theano.config.optimizer = 'fast_run'



np.random.seed(1)

Queries = 1
acquisition_iterations = 380

all_rmse = 0

# We load the boston housing dataset

data = np.loadtxt('boston_housing.txt')


X = data[ :, range(data.shape[ 1 ] - 1) ]
y = data[ :, data.shape[ 1 ] - 1 ]

#X.shape = (506, 13)
#y.shape = (506,)

permutation = np.asarray(random.sample(range(0,X.shape[0]), X.shape[0]))


# size_train = np.round(X.shape[ 0 ] * 0.9)
# index_train = permutation[ 0 : size_train ]
# index_test = permutation[ size_train : ]

# X_train_All = X[ index_train, : ]
# y_train_All = y[ index_train ]
# X_test = X[ index_test, : ]
# y_test = y[ index_test ]


# #X_train_All.shape = (455, 13)
# #y_train_All.shape = (455,)
# #X_test.shape = (51,13)

# X_train = X_train_All[0:20, :]
# y_train = y_train_All[0:20]

# X_pool = X_train_All[20:, :]
# y_pool = y_train_All[20:]


#data splits used by Miguel for AL experiments
size_train = 20
size_test = 100
index_train = permutation[ 0 : size_train ]
index_test = permutation[  size_train:size_test+size_train  ]
index_pool = permutation[ size_test+size_train : ]

X_train = X[ index_train, :  ]
y_train = y[ index_train   ]

X_test = X[ index_test, :   ]
y_test = y[ index_test  ]


X_pool = X[ index_pool, :   ]
y_pool = y[ index_pool  ]



print('Starting with training data size of ', X_train.shape[0])
print('Points in the Pool Set', X_pool.shape[0])



# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

#n_hidden_units = 50


#number of hidden units used for AL experiments by Miguel:
n_hidden_units = 10

net = PBP_net.PBP_net(X_train, y_train,
    [ n_hidden_units, n_hidden_units ], normalize = True, n_epochs = 40)


# We make predictions for the test set
m, v, v_noise = net.predict(X_test)

# We compute the test RMSE
rmse = np.sqrt(np.mean((y_test - m)**2))

# We compute the test log-likelihood
test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - 0.5 * (y_test - m)**2 / (v + v_noise))

print rmse
print test_ll

all_rmse = rmse


for i in range(acquisition_iterations):

	print('Acquisition Iteration', i)

	#randomly query points from the pool set given the number of queries made
	x_pool_index = np.asarray(random.sample(range(0, X_pool.shape[0]), Queries))

	Pooled_X = X_pool[x_pool_index, :]
	Pooled_Y = y_pool[x_pool_index]

	X_pool = np.delete(X_pool, (x_pool_index), axis=0)
	y_pool = np.delete(y_pool, (x_pool_index), axis=0)

	X_train = np.concatenate((X_train, Pooled_X), axis=0)
	y_train = np.concatenate((y_train, Pooled_Y), axis=0)

	n_hidden_units = 10
	net = PBP_net.PBP_net(X_train, y_train, [ n_hidden_units, n_hidden_units ], normalize = True, n_epochs = 40)

	m, v, v_noise = net.predict(X_test)
	
	rmse = np.sqrt(np.mean((y_test - m)**2))

	test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - 0.5 * (y_test - m)**2 / (v + v_noise))

	all_rmse = np.append(all_rmse, rmse)

	print('RMSE at Acquisition Iteration ', rmse)
	print('Predictive Log Likelihood at Acquisition Iteration', test_ll)


print('All RMSE:', all_rmse)

np.save('Random_Acquisition_All_RMSE_' + 'iterations_' + str(acquisition_iterations) + '.npy', all_rmse )
stop = timeit.default_timer()
Time = stop - start
print('Total Run Time', Time)
np.save('Total Time PBP Random.npy', Time)





