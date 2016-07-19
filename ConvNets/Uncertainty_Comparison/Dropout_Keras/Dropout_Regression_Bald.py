
from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K  
import random
import scipy.io
import matplotlib.pyplot as plt
from keras.regularizers import l2, activity_l2
import math
import numpy as np
import sys
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.core import Dense, Dropout, Activation, Flatten


np.random.seed(1)

Queries = 1
acquisition_iterations = 380

all_rmse = 0

batch_size = 16
dropout_iterations = 100

Experiments = 1
Experiments_All_RMSE = np.zeros(shape=(acquisition_iterations+1))


for e in range(Experiments):

	print('Experiment Number ', e)


	data = np.loadtxt('boston_housing.txt')


	X = data[ :, range(data.shape[ 1 ] - 1) ]
	y = data[ :, data.shape[ 1 ] - 1 ]

	permutation = np.asarray(random.sample(range(0,X.shape[0]), X.shape[0]))

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

	#normalise the dataset - X_train, y_train and X_test
	std_X_train = np.std(X_train, 0)
	std_X_train[ std_X_train == 0  ] = 1
	mean_X_train = np.mean(X_train, 0)
	mean_y_train = np.mean(y_train)
	std_y_train = np.std(y_train)
	X_train = (X_train - mean_X_train) / std_X_train
	y_train_normalized = (y_train - mean_y_train ) / std_y_train
	X_test = (X_test - mean_X_train) / std_X_train

	model = Sequential()
	model.add(Dense(32, input_dim=13, init='normal', activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(13, init='normal', activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(6, init='normal', activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1, init='normal'))

	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	model.fit(X_train, y_train, nb_epoch=100, batch_size=batch_size)

	y_predicted = model.predict(X_test,batch_size=batch_size, verbose=1)

	rmse = np.sqrt(np.mean((y_test - y_predicted)**2))

	all_rmse = rmse

	print('Starting Active Learning Experiments')

	for i in range(acquisition_iterations):

		print('Acquisition Iteration', i)

		All_Dropout_Scores = np.zeros(shape=(X_pool.shape[0], 1))

		for d in range(dropout_iterations):
			dropout_score = model.predict_stochastic(X_pool,batch_size=batch_size, verbose=1)
			All_Dropout_Scores = np.append(All_Dropout_Scores, dropout_score, axis=1)


		Variance = np.zeros(shape=(All_Dropout_Scores.shape[0]))
		Mean = np.zeros(shape=(All_Dropout_Scores.shape[0]))
		for j in range(All_Dropout_Scores.shape[0]):
			L = All_Dropout_Scores[j, :]
			L_var = np.var(L)
			L_mean = np.mean(L)
			Variance[j] = L_var
			Mean[j] = L_mean


		#select next x with highest predictive variance
		v_sort = Variance.flatten()
		x_pool_index = v_sort.argsort()[-Queries:][::-1]

		Pooled_X = X_pool[x_pool_index, :]
		Pooled_Y = y_pool[x_pool_index]

		X_pool = np.delete(X_pool, (x_pool_index), axis=0)
		y_pool = np.delete(y_pool, (x_pool_index), axis=0)

		X_train = np.concatenate((X_train, Pooled_X), axis=0)
		y_train = np.concatenate((y_train, Pooled_Y), axis=0)

		model.fit(X_train, y_train, nb_epoch=100, batch_size=batch_size)

		y_predicted = model.predict(X_test,batch_size=batch_size, verbose=1)

		rmse = np.sqrt(np.mean((y_test - y_predicted)**2))

		all_rmse = np.append(all_rmse, rmse)

		print('RMSE at Acquisition Iteration ', rmse)
		
		print('All RMSE:', all_rmse)


	Experiments_All_RMSE = Experiments_All_RMSE + all_rmse

	np.save('Dropout_BALD_Acquisition_All_RMSE_' + 'iterations_' + str(acquisition_iterations) + '_Experiment_' + str(e) +'.npy', all_rmse )



print('Saving Average RMSE Over Experiments')

Average_RMSE = np.divide(Experiments_All_RMSE, Experiments)

print('Average RMSE', Average_RMSE)

np.save('Averaged__Dropout_BALD.npy', Average_RMSE)









