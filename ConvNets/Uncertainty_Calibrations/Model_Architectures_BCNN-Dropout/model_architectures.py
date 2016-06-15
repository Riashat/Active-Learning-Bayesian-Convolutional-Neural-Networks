from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
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


# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3




def VGG_Net():

	#model from: #http://www.robots.ox.ac.uk/~vgg/research/very_deep/

	print('VGGNet Architecture')
	
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






def LeNet5():

	#taken from: https://www.microway.com/hpc-tech-tips/keras-theano-deep-learning-frameworks/
	print('LeNet-5 Architecture')

	model = Sequential()

	model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(16, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(120, 1, 1, border_mode='valid', input_shape = (1, img_rows, img_cols)))

	model.add(Flatten())
	model.add(Dense(84, activation='relu'))
	model.add(Dense(nb_classes))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))





def AlexNet():

	print('AlexNet Architecture')
	model = Sequential()

	#layer 1
	model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	#layer 2
	model.add(Convolution2D(256, 5, 5, border_mode='same', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	#layer 3
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, border_mode='same', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))

	#layer 4
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(1024, 3, 3, border_mode='same', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))

	#layer 5
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(1024, 3, 3, border_mode='same', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	#layer 6
	model.add(Flatten())
	model.add(Dense(3072, init='glorot_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	#layer 7
	model.add(Dense(4096, init='glorot_normal'))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	#layer 8
	model.add(Dense(nb_classes, init='glorot_normal'))
	model.add(Activation('softmax'))






def GoogLeNet():

	#taken from: http://euler.stat.yale.edu/~tba3/stat665/lectures/lec18/notebook18.html

	print('GoogLeNet Architecture')
	model = Sequential()

	#layer 1
	model.add(Convolution2D(64, 1, 1, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(96, 1, 1))
	model.add(Activation('relu'))
	model.add(Convolution2D(16, 1, 1, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	#layer 2
	model.add(Convolution2D(128, 3, 3, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 1, 1, border_mode='valid', input_shape = (1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Flatten())

	#output layer
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(nb_classes, activation='softmax'))





def VGG_16():

	#taken from: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
	print('VGG-16 Architecture')
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))






def VGG_19():
	print('VGG19 CNN Architecture')
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, nb_conv, nb_conv, activation='relu'))	
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))	
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))	
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))






def VGG_8():
	print('VGG8 CNN Architecture')
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, nb_conv, nb_conv, activation='relu'))
	model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))

