
import math

import numpy as np

import sys
sys.path.append('PBP_net/')
import PBP_net

import random


import timeit


import theano


np.random.seed(1)

Queries = 1
acquisition_iterations = 380

all_rmse = 0


Experiments = 5
Experiments_All_RMSE = np.zeros(shape=(acquisition_iterations+1))


for e in range(Experiments):

	print('Experiment Number ', e)


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

		#Bald Acquisition Function
		n_hidden_units = 10
		net = PBP_net.PBP_net(X_train, y_train, [ n_hidden_units, n_hidden_units ], normalize = True, n_epochs = 40)
		m_pool, v_pool, v_noise_pool = net.predict(X_pool)

		#select next x with highest predictive variance
		v_sort = v_pool.flatten()
		x_pool_index = v_sort.argsort()[-Queries:][::-1]

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
	
	Experiments_All_RMSE = Experiments_All_RMSE + all_rmse

	np.save('PBP_BALD_Acquisition_All_RMSE_' + 'iterations_' + str(acquisition_iterations) + '_Experiment_' + str(e) +'.npy', all_rmse )



print('Saving Average RMSE Over Experiments')

Average_RMSE = np.divide(Experiments_All_RMSE, Experiments)

np.save('Averaged__PBP_BALD.npy', Average_RMSE)




