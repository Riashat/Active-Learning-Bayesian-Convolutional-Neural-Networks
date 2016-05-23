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

train_points = 500
test_points = 100

# print ('Using poritions of the training and test sets')
X_train = X_train_All[0:train_points, 0:3,0:32,0:32]
y_train = y_train_All[0:train_points, :]

X_test = X_test[0:test_points,0:3,0:32,0:32]
y_test = y_test[0:test_points,:]
#pool of training data points

pool_count=50000
X_Pool = X_train_All[1001:pool_count, 0:3, 0:32, 0:32]
Y_oracle = y_train_All[1001:pool_count, :]


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

number_acquisitions_iterations = 5
N = 1   	#number of samples

all_accuracy = 0

#for number of acquisitions:
for i in range(number_acquisitions_iterations):

	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	All_Y_Probability = np.zeros(shape=(X_train.shape[0], nb_classes))
	All_Entropy_P_Y = np.zeros(shape=(X_train.shape[0], nb_classes))

	for j in range(N):

		model = Sequential()
		model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
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

		model.add(Convolution2D(128, 3, 3, border_mode='same'))
		model.add(Activation('relu'))
		model.add(Convolution2D(128, 3, 3))
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
		X_test /= 252

		model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)

		Y_Probability = model.predict_proba(X_train, batch_size=batch_size, verbose=1)

		All_Y_Probability = All_Y_Probability + Y_Probability

		
		log_P_Y = np.log2(Y_Probability)
		Entropy_P_Y = - np.multiply(Y_Probability, log_P_Y)
		All_Entropy_P_Y = All_Entropy_P_Y + Entropy_P_Y



	# calculating f(x)
	MC_All_Y_Probability = np.divide(All_Y_Probability, N)
	Class_Log_Probability = np.log2(MC_All_Y_Probability)
	Entropy_Each_Cell = - np.multiply(All_Y_Probability, Class_Log_Probability)
	F_X = np.sum(Entropy_Each_Cell, axis=1)


	#calculating g(x)
	G_X_Entropy = np.sum(All_Entropy_P_Y, axis=1)
	G_X = np.divide(G_X_Entropy, N)

	U_X = F_X - G_X

	P = 1   # number of aquisitions per iteration
	x_pool_index = U_X.argsort()[-P:][::-1]

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

np.savetxt("Bayesian Active Learning Accuracy Values.csv", all_accuracy, delimiter=",")