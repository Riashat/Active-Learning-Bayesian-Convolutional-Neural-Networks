import math
import numpy as np
import sys
import theano
sys.path.append('../code/')
import AEPDGP_net
import matplotlib.pyplot as plt
import time
import random

import timeit


np.random.seed(1234)


# number of GP layers
nolayers = 2
# number of hidden dimension in intermediate hidden layers
n_hiddens = [2]
# number of inducing points per layer
M = 20
n_pseudos = [M for _ in range(nolayers)]


#no_iterations = 1000

no_iterations = 1000

no_points_per_mb = 50

Queries = 1
acquisition_iterations = 380

all_rmse = 0



Experiments = 5
Experiments_All_RMSE = np.zeros(shape=(acquisition_iterations+1))


for e in range(Experiments):

	print('Experiment Number ', e)

	# We load the boston housing dataset
	data = np.loadtxt('boston_housing.txt')


	X = data[ :, range(data.shape[ 1 ] - 1) ]
	y = data[ :, data.shape[ 1 ] - 1 ]

	permutation = np.asarray(random.sample(range(0,X.shape[0]), X.shape[0]))

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



	# We construct the network
	net = AEPDGP_net.AEPDGP_net(X_train, y_train, n_hiddens, n_pseudos)
	t0 = time.time()
	# train
	test_nll, test_rms, energy = net.train(no_iterations=no_iterations,
	                               no_points_per_mb=no_points_per_mb,
	                               lrate=0.02)

	# We make predictions for the test set
	m, v = net.predict(X_test)

	# We compute the test RMSE
	rmse = np.sqrt(np.mean((y_test - m)**2))

	# We compute the test log-likelihood
	test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v)) - \
	    0.5 * (y_test - m)**2 / (v))

	print 'test rmse: ', rmse
	print 'test log-likelihood: ', test_ll


	all_rmse = rmse

	for i in range(acquisition_iterations):

		print('Acquisition Iterations', acquisition_iterations)

		x_pool_index = np.asarray(random.sample(range(0, X_pool.shape[0]), Queries))

		Pooled_X = X_pool[x_pool_index, :]
		Pooled_Y = y_pool[x_pool_index]

		X_pool = np.delete(X_pool, (x_pool_index), axis=0)
		y_pool = np.delete(y_pool, (x_pool_index), axis=0)


		X_train = np.concatenate((X_train, Pooled_X), axis=0)
		y_train = np.concatenate((y_train, Pooled_Y), axis=0)

		# We construct the network
		net = AEPDGP_net.AEPDGP_net(X_train, y_train, n_hiddens, n_pseudos)

		# train
		test_nll, test_rms, energy = net.train(no_iterations=no_iterations,
		                               no_points_per_mb=no_points_per_mb,
		                               lrate=0.02)

		# We make predictions for the test set
		m, v = net.predict(X_test)

		# We compute the test RMSE
		rmse = np.sqrt(np.mean((y_test - m)**2))

		# We compute the test log-likelihood
		test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v)) - \
		    0.5 * (y_test - m)**2 / (v))

		print 'test rmse: ', rmse
		print 'test log-likelihood: ', test_ll

		all_rmse = np.append(all_rmse, rmse)


	Experiments_All_RMSE = Experiments_All_RMSE + all_rmse

	np.save('AEPDGP_Random_Acquisition_All_RMSE_' + 'iterations_' + str(acquisition_iterations) + '_Experiment_' + str(e) +'.npy', all_rmse )


print('Saving Average RMSE Over Experiments')

Average_RMSE = np.divide(Experiments_All_RMSE, Experiments)

np.save('Averaged__AEPDGP_Random.npy', Average_RMSE)




