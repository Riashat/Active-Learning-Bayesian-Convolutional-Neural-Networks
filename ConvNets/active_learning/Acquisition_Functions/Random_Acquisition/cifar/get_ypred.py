from __future__ import print_function
from keras.datasets import cifar10
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
import scipy.io

batch_size = 32
nb_classes = 10
nb_epoch = 1
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train_All, y_train_All), (X_test, y_test) = cifar10.load_data()

print('Original size of the cifar10 dataset')
print('X_train shape:', X_train_All.shape)
print('y_train shape:', y_train_All.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


X_train = X_train_All[0:2000, 0:3,0:32,0:32]
y_train = y_train_All[0:2000, :]


#pool of training data points
X_Pool = X_train_All[2001:49000, 0:3, 0:32, 0:32]
# no labels available for the Pool Training Points

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


print('BUILDING THE Bayesian ConvNet Model')

model = Sequential()


model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))   #using relu activation function
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Flatten used to Flatten the input : 64*32*32 to the product of these (65536)

# Dense: just the regular fully connected NN layer
# Dense(512) means - output arrays of shape (*, 512)
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# let's train the model using SGD + momentum (how original).
# SGD from Optimizers
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Once your model looks good, configure its learning process with .compile():
# also using the Objective function "categorical_crossentropy" or can use "mean_squared_error" here
model.compile(loss='categorical_crossentropy', optimizer=sgd)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('TRAINING THE MODEL')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)


print('Test Time Dropout')
print('Y Predictions for the Pool Data')
# make Y_Predictions for the POOL of Unlabelled Data
# PREDICT - does dropout at test time
# REPEAT THIS 5 times - and get average predictions and standard deviations
score = model.predict(X_Pool,batch_size=batch_size, verbose=1)		#score is a numpy array

scipy.io.savemat('Score3.mat', dict(score3 = score))








