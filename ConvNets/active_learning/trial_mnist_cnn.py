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

batch_size = 128
nb_classes = 10
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets
(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()

X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)


X_valid = X_train_All[5000:7000, :, :, :]
y_valid = y_train_All[5000:7000]

X_train = X_train_All[0:5000, :, :, :]
y_train = y_train_All[0:5000]

X_Pool = X_train_All[7000:60000, :, :, :]
y_Pool = y_train_All[7000:60000]

X_test = X_test[0:2000, :, :, :]
y_test = y_test[0:2000]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')


N = 7000
R = np.asarray(random.sample(range(0, 52999), N))

Pooled_X = X_Pool[R, :, :, :]
Pooled_Y = y_Pool[R]

X_train = np.concatenate((X_train, Pooled_X), axis=0)
y_train = np.concatenate((y_train, Pooled_Y), axis=0)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_valid = X_valid.astype('float32')
X_Pool = X_Pool.astype('float32')
X_train /= 255
X_valid /= 255
X_Pool /= 255
X_test /= 255

print('After random acquisitions')
print(X_train.shape[0], 'train samples after acquisition')
print(X_test.shape[0], 'test samples')
print(X_valid.shape[0], 'validation samples')
print(X_Pool.shape[0], 'pool samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)
Y_Pool = np_utils.to_categorical(y_Pool, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
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

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))


score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)


print('Test score:', score[0])
print('Test accuracy:', score[1])
