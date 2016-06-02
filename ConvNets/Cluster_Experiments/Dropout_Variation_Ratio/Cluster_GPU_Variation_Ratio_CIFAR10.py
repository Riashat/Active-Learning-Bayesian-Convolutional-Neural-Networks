from __future__ import print_function
from keras.datasets import cifar10
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
from scipy.stats import mode

Experiments = 3

batch_size = 128
nb_classes = 10


#use a large number of epochs
nb_epoch = 30


# input image dimensions
img_rows, img_cols = 32, 32

# the CIFAR10 images are RGB
img_channels = 3

score=0
all_accuracy = 0
acquisition_iterations = 300

#use a large number of dropout iterations
dropout_iterations = 100

Queries = 10

Experiments_All_Accuracy = np.zeros(shape=(acquisition_iterations+1))

for e in range(Experiments):

	print('Experiment Number ', e)



	# the data, shuffled and split between train and test sets
	(X_train_All, y_train_All), (X_test, y_test) = cifar10.load_data()

	#after 50 iterations with 10 pools - we have 500 pooled points - use validation set outside of this
	X_valid = X_train_All[4000:4150, 0:3, 0:32, 0:32]
	y_valid = y_train_All[4000:4150, 0]

	X_train = X_train_All[0:200, 0:3, 0:32, 0:32]
	y_train = y_train_All[0:200, 0]

	X_Pool = X_train_All[5000:15000, 0:3, 0:32, 0:32]
	y_Pool = y_train_All[5000:15000, 0]



	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')


	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_valid = X_valid.astype('float32')
	X_Pool = X_Pool.astype('float32')
	X_train /= 255
	X_valid /= 255
	X_Pool /= 255
	X_test /= 255

	Y_test = np_utils.to_categorical(y_test, nb_classes)
	Y_valid = np_utils.to_categorical(y_valid, nb_classes)
	Y_Pool = np_utils.to_categorical(y_Pool, nb_classes)


	#loss values in each experiment
	Pool_Valid_Loss = np.zeros(shape=(nb_epoch, 1)) 	#row - no.of epochs, col (gets appended) - no of pooling
	Pool_Train_Loss = np.zeros(shape=(nb_epoch, 1)) 
	x_pool_All = np.zeros(shape=(1))

	Y_train = np_utils.to_categorical(y_train, nb_classes)

	print('Training Model Without Acquisitions in Experiment', e)



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

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))
	Train_Result_Optimizer = hist.history
	Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	Train_Loss = np.array([Train_Loss]).T
	Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
	Valid_Loss = np.asarray([Valid_Loss]).T

	Pool_Train_Loss = Train_Loss
	Pool_Valid_Loss = Valid_Loss

	print('Evaluating Test Accuracy Without Acquisition')
	score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

	all_accuracy = acc

	print('Starting Active Learning in Experiment ', e)



	for i in range(acquisition_iterations):
		print('POOLING ITERATION', i)


		All_Dropout_Classes = np.zeros(shape=(X_Pool.shape[0],1))
		print('Use trained model for test time dropout')

		for d in range(dropout_iterations):
			print ('Dropout Iteration', d)
			dropout_classes = model.predict_classes_stochastic(X_Pool,batch_size=batch_size, verbose=1)
			dropout_classes = np.array([dropout_classes]).T
			#np.save('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/BCNN_Maximal_Uncertainty/Variation_Ratio/Dropout_Scores/'+'Dropout_Score_'+str(d)+'.npy',dropout_classes)
			All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

		Variation = np.zeros(shape=(X_Pool.shape[0]))

		for t in range(X_Pool.shape[0]):
			L = np.array([0])
			for d_iter in range(dropout_iterations):
				L = np.append(L, All_Dropout_Classes[t, d_iter+1])						
			Predicted_Class, Mode = mode(L[1:])
			v = np.array(  [1 - Mode/float(dropout_iterations)])
			Variation[t] = v


		a_1d = Variation.flatten()
		x_pool_index = a_1d.argsort()[-Queries:][::-1]

		#store all the pooled images indexes
		x_pool_All = np.append(x_pool_All, x_pool_index)

		#saving pooled images
		# for im in range(4):
		# 	Image = X_Pool[x_pool_index[im], :, :, :]
		# 	img = Image.reshape((28,28))
		# 	sp.misc.imsave(''+'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)


		Pooled_X = X_Pool[x_pool_index, 0:3,0:32,0:32]
		Pooled_Y = y_Pool[x_pool_index]	

		delete_std = np.delete(Variation, (x_pool_index), axis=0)
		delete_Pool_X = np.delete(X_Pool, (x_pool_index), axis=0)
		delete_Pool_Y = np.delete(y_Pool, (x_pool_index), axis=0)

		print('Acquised Points added to training set')

		X_train = np.concatenate((X_train, Pooled_X), axis=0)
		y_train = np.concatenate((y_train, Pooled_Y), axis=0)

		print('Train Model with pooled points')


		# convert class vectors to binary class matrices
		Y_train = np_utils.to_categorical(y_train, nb_classes)


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

		model.compile(loss='categorical_crossentropy', optimizer='adam')
		hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))
		Train_Result_Optimizer = hist.history
		Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
		Train_Loss = np.array([Train_Loss]).T
		Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
		Valid_Loss = np.asarray([Valid_Loss]).T


		#Accumulate the training and validation/test loss after every pooling iteration - for plotting
		Pool_Valid_Loss = np.append(Pool_Valid_Loss, Valid_Loss, axis=1)
		Pool_Train_Loss = np.append(Pool_Train_Loss, Train_Loss, axis=1)	

		print('Evaluate Model Test Accuracy with pooled points')

		score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
		print('Test score:', score)
		print('Test accuracy:', acc)
		all_accuracy = np.append(all_accuracy, acc)

		print('Use this trained model with pooled points for Dropout again')


	print('Storing Accuracy Values over experiments')
	Experiments_All_Accuracy = Experiments_All_Accuracy + all_accuracy


	print('Saving Results Per Experiment')
	np.save(''+'CIFAR10_All_Train_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Loss)
	np.save(''+ 'CIFAR10_All_Valid_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Loss)
	np.save(''+'CIFAR10_All_Pooled_Image_Index_'+ 'Experiment_' + str(e) + '.npy', x_pool_All)
	np.save(''+ 'CIFAR10_All_Accuracy_Results_'+ 'Experiment_' + str(e) + '.npy', all_accuracy)

print('Saving Average Accuracy Over Experiments')

Average_Accuracy = np.divide(Experiments_All_Accuracy, Experiments)

np.save(''+'CIFAR10_Average_Accuracy'+'.npy', Average_Accuracy)








