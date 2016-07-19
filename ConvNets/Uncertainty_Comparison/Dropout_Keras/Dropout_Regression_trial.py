
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



batch_size = 16
dropout_iterations = 5

data = np.loadtxt('boston_housing.txt')


X = data[ :, range(data.shape[ 1 ] - 1) ]
y = data[ :, data.shape[ 1 ] - 1 ]

permutation = np.asarray(random.sample(range(0,X.shape[0]), X.shape[0]))

size_train = 100
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


#normalise the dataset - X_train, y_train and X_test
std_X_train = np.std(X_train, 0)
std_X_train[ std_X_train == 0  ] = 1
mean_X_train = np.mean(X_train, 0)

mean_y_train = np.mean(y_train)
std_y_train = np.std(y_train)


X_train = (X_train - mean_X_train) / std_X_train

y_train_normalized = (y_train - mean_y_train ) / std_y_train

X_test = (X_test - mean_X_train) / std_X_train


#build the keras model for regression
model = Sequential()
# model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
# model.add(Dense(6, init='normal', activation='relu'))
# model.add(Dense(1, init='normal'))

# model.compile(loss='mean_absolute_error', optimizer='rmsprop')

# model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
# score = model.evaluate(X_test, y_test, batch_size=16)


#this one works as well
model.add(Dense(32, input_dim=13, init='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(13, init='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(6, init='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, init='normal'))


model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=100, batch_size=16)

All_Dropout_Scores = np.zeros(shape=(100, 1))
for d in range(dropout_iterations):
	dropout_score = model.predict_stochastic(X_test,batch_size=batch_size, verbose=1)
	All_Dropout_Scores = np.append(All_Dropout_Scores, dropout_score, axis=1)



Variance = np.zeros(shape=(All_Dropout_Scores.shape[0]))
Mean = np.zeros(shape=(All_Dropout_Scores.shape[0]))
for j in range(All_Dropout_Scores.shape[0]):
	L = All_Dropout_Scores[j, :]
	L_var = np.var(L)
	L_mean = np.mean(L)
	Variance[j] = L_var
	Mean[j] = L_mean



y_predicted = model.predict(X_test,batch_size=batch_size, verbose=1)

rmse = np.sqrt(np.mean((y_test - y_predicted)**2))

rmse2 = np.sqrt(np.mean((y_test - Mean)**2))



print(rmse)
print(rmse2)


#9.551411151
#6.56374192961



# dropout_score1 = model.predict_stochastic(X_test,batch_size=batch_size, verbose=1)
# print('Dropout Score 1', dropout_score1)

# dropout_score2 = model.predict_stochastic(X_test,batch_size=batch_size, verbose=1)
# print('Dropout Score 2', dropout_score2)

# dropout_score3 = model.predict_stochastic(X_test,batch_size=batch_size, verbose=1)
# print('Dropout Score 3', dropout_score3)

# dropout_score4 = model.predict_stochastic(X_test,batch_size=batch_size, verbose=1)
# print('Dropout Score 4', dropout_score4)

# score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)




