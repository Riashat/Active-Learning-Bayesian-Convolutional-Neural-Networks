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
import random
import scipy.io
import matplotlib.pyplot as plt


batch_size = 32
nb_classes = 10
nb_epoch = 5
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train_All, y_train_All), (X_test, y_test) = cifar10.load_data()

# print('Original size of the cifar10 dataset')
# print('X_train shape:', X_train_All.shape)
# print('y_train shape:', y_train_All.shape)
# print('X_test shape:', X_test.shape)
# print('y_test shape:', y_test.shape)

#Original Training Data
X_train = X_train_All[0:1000, 0:3,0:32,0:32]
y_train = y_train_All[0:1000, :]

X_test = X_test[0:500,0:3,0:32,0:32]
y_test = y_test[0:500,:]

#pool of training data points
# Training data points divided into training and pool set
X_Pool = X_train_All[3000:50000, 0:3, 0:32, 0:32]
Y_Oracle = y_train_All[3000:50000, :]

score=0
accuracy = 0
Number_Queries = 7
data_size = X_train.shape[0]
acquisition_iterations = 3

All_Train_Loss = np.zeros(shape=(nb_epoch, 1))

# i denotes the number of times to concatenate or perform acquisition from the pool data
for i in range(acquisition_iterations):

	print('POOLING ITERATION NUMBER', i)

	# randomly selecting a pool data point
	R = np.asarray(random.sample(range(2001,47000), 1000))	

	# Remaining_Pool_Points = np.delete(X_Pool, R, 0)		
	Pooled_X = X_Pool[R, :]

	# get the true labels from the oracle
	Pooled_Y = Y_Oracle[R, :]

	X_train = np.concatenate((X_train, Pooled_X), axis=0)
	y_train = np.concatenate((y_train, Pooled_Y), axis=0)

	# convert class vectors to binary class matrices
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
	# X_Pool = Remaining_Pool_Points


	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
	Train_Result_Optimizer = hist.history
	Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	Train_Loss = np.array([Train_Loss]).T

	# # plot training loss
	# plt.plot(Train_Loss)
	# plt.ylabel('Training Loss')
	# plt.xlabel('Number of Epochs')
	# plt.show()

	#monitoring optimisation with number of acquisitions
	#storing all training loss with number of acquisitions
	All_Train_Loss = np.concatenate((All_Train_Loss, Train_Loss), axis=1)





plt.plot(All_Train_Loss)
plt.ylabel('Training Loss with Acquisitions')
plt.xlabel('Number of Epochs')
plt.show()



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_Pool = Remaining_Pool_Points

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
Train_Result_Optimizer = hist.history
Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
Train_Loss = np.array([Train_Loss]).T

# plt.plot(Train_Loss)
# plt.ylabel('Training Loss')
# plt.xlabel('Number of Epochs')
# plt.show()


evaluation_score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test Accuracy:', evaluation_score[1])

eval_score = evaluation_score[0]
eval_accuracy = evaluation_score[1]

score = np.append(score, eval_score)
accuracy = np.append(accuracy, eval_accuracy)
data_size = np.append(data_size, X_train.shape[0])

print ('Training Data Size', data_size)
print('All Accuracy Values:', accuracy)


# np.savetxt("Acquisition_Scores.csv", score, delimiter=",")
np.savetxt("Acquisition_Accuracy.csv", accuracy, delimiter=",")



