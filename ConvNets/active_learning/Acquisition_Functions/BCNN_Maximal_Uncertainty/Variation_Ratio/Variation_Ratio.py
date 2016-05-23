from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K  
import random
import scipy.io
import matplotlib.pyplot as plt
batch_size = 128
nb_classes = 10
nb_epoch = 2

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


X_train_Seg = X_train[0:12000, :, :, :]
Y_train_Seg = Y_train[0:12000, :]
X_test = X_test[0:2000, :, :, :]
Y_test = Y_test[0:2000, :]

X_Pool = X_train_Seg[4000:5000, :, :, :]
Y_Oracle = Y_train_Seg[4000:5000, :] 

X_train = X_train_Seg[0:2000, :, :, :]
Y_train = Y_train_Seg[0:2000, :]

X_valid = X_train_Seg[10000:12000, :, :, :]
Y_valid = Y_train_Seg[10000:12000, :]

score=0
loss = 0
Pooled_Y = np.zeros(1)
Queries = 1000
score=0
accuracy=0
sklearn_accuracy=0
pool_iterations = 1
All_Train_Loss = np.zeros(shape=(nb_epoch, 1))

dropout_iterations = 5



for i in range(pool_iterations):

	print('POOLING ITERATION', i)

	Y_train = np_utils.to_categorical(Y_train, nb_classes)
	Y_test = np_utils.to_categorical(Y_test, nb_classes)
	Y_valid = np_utils.to_categorical(Y_valid, nb_classes)
	Y_Oracle = np_utils.to_categorical(Y_Oracle, nb_classes)



	model = Sequential()
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_valid = X_valid.astype('float32')
	X_Pool = X_Pool.astype('float32')
	X_train /= 255
	X_test /= 255
	X_valid /= 255
	X_Pool /= 255

	model.compile(loss='categorical_crossentropy', optimizer='adadelta')
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))
	Train_Result_Optimizer = hist.history
	Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	Train_Loss = np.array([Train_Loss]).T
	Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
	Valid_Loss = np.asarray([Valid_Loss]).T

	Y_Pool_Pred = model.predict_classes(X_Pool, batch_size=batch_size, verbose=1)


