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
import sklearn
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt

batch_size = 32
nb_classes = 10
nb_epoch = 15
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

train_points = 2000
test_points = 1000

# print ('Using poritions of the training and test sets')
X_train = X_train_All[0:train_points, 0:3,0:32,0:32]
y_train = y_train_All[0:train_points, :]

X_test = X_test[0:test_points,0:3,0:32,0:32]
y_test = y_test[0:test_points,:]


pool_count = 5000
#pool of training data points
X_Pool = X_train_All[2000:pool_count, 0:3, 0:32, 0:32]
Y_oracle = y_train_All[2000:pool_count, :]


score=0
accuracy=0
all_accuracy = 0
iterations = 12     # this should be large value - denotes number of acquisitions to make using highest entropy
# to get N max values - for N acquisitions
N = 100

All_Train_Loss = np.zeros(shape=(nb_epoch, 1))


for i in range(iterations):

	print('POOLING ITERATION', i)

	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

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

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255

	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
	Train_Result_Optimizer = hist.history
	Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	Train_Loss = np.array([Train_Loss]).T

	All_Train_Loss = np.concatenate((All_Train_Loss, Train_Loss), axis=1)

	Class_Probability = model.predict_proba(X_Pool, batch_size=batch_size, verbose=1)

	# alternatively, take the max of Class_Probability for each datapoint
	#we get P_i
	# compute the entropy H(x) for each datapoint with P_i


	Class_Log_Probability = np.log2(Class_Probability)
	Entropy_Each_Cell = - np.multiply(Class_Probability, Class_Log_Probability)

	Entropy = np.sum(Entropy_Each_Cell, axis=1)	# summing across rows of the array

	#x_pool_index = 	np.unravel_index(Entropy.argmax(), Entropy.shape)	#for finding the maximum value np.amax(Entropy)
	x_pool_index = Entropy.argsort()[-N:][::-1]

	Pooled_X = X_Pool[x_pool_index, 0:3,0:32,0:32]
	Pooled_Y = Y_oracle[x_pool_index,:]		# true label from the oracle

	X_train = np.concatenate((X_train, Pooled_X), axis=0)
	y_train = np.concatenate((y_train, Pooled_Y), axis=0)

	#delete the currently pooled points from the pool set
	X_Pool = np.delete(X_Pool, x_pool_index, 0)


# Finally, after number of iterations of acquisitions, we have X_train and y_train

print('SIZE OF TRAINING DATA AFTER ACQUISITIONS', X_train.shape)


print('Training the overall model after all acquisitions')
# Fit the model again, with all the acquisted points from the pool data
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


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


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)


print('TEST THE MODEL ACCURACY')
# Compute the test error and accuracy 
score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

print('Test score:', score)
print('Test accuracy:', acc)

all_accuracy = np.append(all_accuracy, acc)

np.savetxt("Highest Entropy Accuracy Values.csv", all_accuracy, delimiter=",")


# plt.plot(All_Train_Loss)
# plt.ylabel('Training Loss with Acquisitions')
# plt.xlabel('Number of Epochs')
# plt.show()
