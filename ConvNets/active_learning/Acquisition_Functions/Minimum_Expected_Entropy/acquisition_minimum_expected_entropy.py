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
import numpy as np
import math

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

train_points = 100
test_points = 50

X_train = X_train_All[0:train_points, 0:3,0:32,0:32]
y_train = y_train_All[0:train_points, :]

X_test = X_test[0:test_points,0:3,0:32,0:32]
y_test = y_test[0:test_points,:]
#pool of training data points

pool_count = 1500
X_Pool = X_train_All[1001:pool_count, 0:3, 0:32, 0:32]
Y_oracle = y_train_All[1001:pool_count, :]

# no labels available for the Pool Training Points

Train_After_Acquisition_X = X_train
Train_After_Acquisition_Y = y_train
Train_After_Acquisition_Y = np_utils.to_categorical(Train_After_Acquisition_Y, nb_classes)

score=0
accuracy=0
sklearn_accuracy=0
all_accuracy = 0
iterations = 3


for i in range(iterations):			# for each round t

	print('ITERATION NUMBER', i)

	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	for pool_point in range(X_Pool.shape[0]):# X_Pool.shape[0]			# for each unlabelled image
		
		for pool_label in range(2): # nb_classes			# for each possible class label
			
			#estimate P(Y_i =j | L)
			# L - labelled points
			Labelled_data_X = X_train
			Labelled_data_Y = Y_train

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

			model.add(Flatten())
			model.add(Dense(512))
			model.add(Activation('relu'))
			model.add(Dropout(0.5))
			model.add(Dense(nb_classes))
			model.add(Activation('softmax'))

			sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
			model.compile(loss='categorical_crossentropy', optimizer=sgd)

			Labelled_data_X = Labelled_data_X.astype('float32')
			X_Pool = X_Pool.astype('float32')
			Labelled_data_X /= 255
			X_Pool /= 255

			model.fit(Labelled_data_X, Labelled_data_Y, batch_size=batch_size, nb_epoch=nb_epoch)

			Class_Probability = model.predict_proba(X_Pool, batch_size=batch_size, verbose=1)

			index = np.array([pool_point])
			i_pool_x = X_Pool[index, 0:3, 0:32, 0:32]
			i_pool_y = Y_oracle[index, :]

			#all other pool points other than current point
			Remaining_Pool_Points = np.delete(X_Pool, pool_point, 0)
			i_pool_y = np_utils.to_categorical(i_pool_y, nb_classes)


			for rm_pool_point in range(Remaining_Pool_Points.shape[0]):	#Remaining_Pool_Points.shape[0]
				for pool_class in range(2):	#nb_classes

					New_Labelled_data_X = np.concatenate((Labelled_data_X, i_pool_x), axis=0)
					New_Labelled_data_Y = np.concatenate((Labelled_data_Y, i_pool_y), axis=0)

					New_Labelled_data_Y	= np_utils.to_categorical(New_Labelled_data_Y, nb_classes)

					#estimate P(Y_k =j | L U (x_i, j))
					# fit a model with the new concatenated labelled data - using the unlabelled image and queried point

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

					model.add(Flatten())
					model.add(Dense(512))
					model.add(Activation('relu'))
					model.add(Dropout(0.5))
					model.add(Dense(nb_classes))
					model.add(Activation('softmax'))

					sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
					model.compile(loss='categorical_crossentropy', optimizer=sgd)

					New_Labelled_data_X = New_Labelled_data_X.astype('float32')
					Remaining_Pool_Points = Remaining_Pool_Points.astype('float32')
					New_Labelled_data_X /= 255
					Remaining_Pool_Points /= 255

					model.fit(New_Labelled_data_X, New_Labelled_data_Y, batch_size=batch_size, nb_epoch=nb_epoch)

					# set of P(Y_k = l | L U (x_i, j)) values0
					P_Y_k = model.predict_proba(Remaining_Pool_Points, batch_size=batch_size, verbose=1)

					log_P_Y_k = np.log2(P_Y_k)
					Entropy_Y_k = - np.multiply(P_Y_k, log_P_Y_k)

					#Entropy_Y_k = np.sum(Entropy_Y_k_Each_Cell, axis=1)	# summing across rows of the array

		# Computing H_x - the expected Entropy			
		H_x = np.array([[     np.dot(   Class_Probability[pool_point , :],  Entropy_Y_k[pool_point, :] )     ]])
		H_x = np.append(H_x, H_x, axis=0)


	# acquise the point with argmin (H_x) and its index value
	acquised_point = np.amin(H_x)
	acquised_point_index = np.where(H_x==H_x.min())[0]
	#acquised_point_index = H_x.argsort()[:1]

	Pooled_X = X_Pool[acquised_point_index, 0:3,0:32,0:32]
	Pooled_Y = Y_oracle[acquised_point_index, :]
	Pooled_Y = np_utils.to_categorical(Pooled_Y, nb_classes)

	# accumulate all the acquised points from X_Pool - accumulate all indices of the acquised points
	# concatenate all the acquised points with the training data
	Train_After_Acquisition_X = np.concatenate(  ( Train_After_Acquisition_X, Pooled_X   ), axis=0   )
	Train_After_Acquisition_Y = np.concatenate(  ( Train_After_Acquisition_Y, Pooled_Y   ), axis=0   )




# Fit the model with all the training data
Train_After_Acquisition_Y = np_utils.to_categorical(Train_After_Acquisition_Y, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

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

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd)

Train_After_Acquisition_X = Train_After_Acquisition_X.astype('float32')
X_test = X_test.astype('float32')
Train_After_Acquisition_X /= 255
X_test /= 255

model.fit(Train_After_Acquisition_X, Train_After_Acquisition_Y, batch_size=batch_size, nb_epoch=nb_epoch)

# Test the model
print('TEST THE MODEL ACCURACY')
# Compute the test error and accuracy 
score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

print('Test score:', score)
print('Test accuracy:', acc)

all_accuracy = np.append(all_accuracy, acc)

np.savetxt("Minimum Expected Error Accuracy Values.csv", all_accuracy, delimiter=",")






print ('DONE')






