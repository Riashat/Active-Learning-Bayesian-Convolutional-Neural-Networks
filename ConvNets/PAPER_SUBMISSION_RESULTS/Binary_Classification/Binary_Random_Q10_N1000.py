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


Experiments = 3

batch_size = 128
nb_classes = 10

#use a large number of epochs
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

score=0
all_accuracy = 0
acquisition_iterations = 98

#use a large number of dropout iterations
dropout_iterations = 100

Queries = 10


Experiments_All_Accuracy = np.zeros(shape=(acquisition_iterations+1))



for e in range(Experiments):

	print('Experiment Number ', e)

	# the data, shuffled and split between tran and test sets
	(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()

	X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

	random_split = np.asarray(random.sample(range(0,X_train_All.shape[0]), X_train_All.shape[0]))

	X_train_All = X_train_All[random_split, :, :, :]
	y_train_All = y_train_All[random_split]

	#Find Binary Images - MBR over Binary Images only - considering only images 2 and 8
	Class_2_Train = np.where(y_train_All==2)[0]
	Class_8_Train = np.where(y_train_All==8)[0]
	y_2 = y_train_All[Class_2_Train]
	X_2 = X_train_All[Class_2_Train, :, :, :]
	y_8 = y_train_All[Class_8_Train]
	X_8 = X_train_All[Class_8_Train, :, :, :]

	X_train_All = np.concatenate((X_2, X_8), axis=0)
	y_train_All = np.concatenate((y_2, y_8), axis=0)


	#defines how many training points to start with
	X_train_All = X_train_All[0:10000, :, :, :]
	y_train_All = y_train_All[0:10000]

	Class_2_Test = np.where(y_test==2)[0]
	Class_8_Test = np.where(y_test==8)[0]
	y_2_test = y_test[Class_2_Test]
	X_2_test = X_test[Class_2_Test, :, :, :]
	y_8_test = y_test[Class_8_Test]
	X_8_test = X_test[Class_8_Test, :, :, :]

	X_test = np.concatenate((X_2_test, X_8_test), axis=0)
	y_test = np.concatenate((y_2_test, y_8_test), axis=0)


	#number of test points
	X_test = X_test[0:5000, :, :, :]
	y_test = y_test[0:5000]


	#use 1000 validation points
	X_valid = X_train_All[2000:3000, :, :, :]
	y_valid = y_train_All[2000:3000]

	X_train = X_train_All[0:20, :, :, :]
	y_train = y_train_All[0:20]

	#use 10000 pool points to start with
	X_Pool = X_train_All[3000:10000, :, :, :]
	y_Pool = y_train_All[3000:10000]


	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')

	print('Distribution of Training Classes:', np.bincount(y_train))
	

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
	Pool_Valid_Loss = np.zeros(shape=(nb_epoch, 1)) 	
	Pool_Train_Loss = np.zeros(shape=(nb_epoch, 1)) 
	Pool_Valid_Acc = np.zeros(shape=(nb_epoch, 1)) 	
	Pool_Train_Acc = np.zeros(shape=(nb_epoch, 1)) 
	x_pool_All = np.zeros(shape=(1))

	Y_train = np_utils.to_categorical(y_train, nb_classes)

	print('Training Model Without Acquisitions in Experiment', e)


	model = Sequential()
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.25))
	
	model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.25))

	c = 3.5
	Weight_Decay = c / float(X_train.shape[0])
	model.add(Flatten())
	model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
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
	Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
	Train_Acc = np.array([Train_Acc]).T
	Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
	Valid_Acc = np.asarray([Valid_Acc]).T


	Pool_Train_Loss = Train_Loss
	Pool_Valid_Loss = Valid_Loss
	Pool_Train_Acc = Train_Acc
	Pool_Valid_Acc = Valid_Acc



	Pool_Train_Loss = Train_Loss
	Pool_Valid_Loss = Valid_Loss

	print('Evaluating Test Accuracy Without Acquisition')
	score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

	all_accuracy = acc

	print('Starting Active Learning in Experiment ', e)



	# i denotes the number of times to concatenate or perform acquisition from the pool data
	for i in range(acquisition_iterations):

		print('POOLING ITERATION NUMBER', i)


		x_pool_index = np.asarray(random.sample(range(0, 6500), Queries))

		Pooled_X = X_Pool[x_pool_index, :, :, :]
		Pooled_Y = y_Pool[x_pool_index]

		x_pool_All = np.append(x_pool_All, x_pool_index)

		# #saving pooled images
		# for im in range(x_pool_index[0:2].shape[0]):
		# 	Image = X_Pool[x_pool_index[im], :, :, :]
		# 	img = Image.reshape((28,28))
		# 	sp.misc.imsave('/home/ri258/Documents/Project/Active-Learning-Deep-Convolutional-Neural-Networks/ConvNets/Cluster_Experiments/Random_Acquisition/Pooled_Images/'+'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)

	
		X_train = np.concatenate((X_train, Pooled_X), axis=0)
		y_train = np.concatenate((y_train, Pooled_Y), axis=0)


		# convert class vectors to binary class matrices
		Y_train = np_utils.to_categorical(y_train, nb_classes)

		model = Sequential()
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
		model.add(Activation('relu'))
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
		model.add(Dropout(0.25))


		c = 10
		Weight_Decay = c / float(X_train.shape[0])
		model.add(Flatten())
		model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
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
		Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
		Train_Acc = np.array([Train_Acc]).T
		Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
		Valid_Acc = np.asarray([Valid_Acc]).T

		#Accumulate the training and validation/test loss after every pooling iteration - for plotting
		Pool_Valid_Loss = np.append(Pool_Valid_Loss, Valid_Loss, axis=1)
		Pool_Train_Loss = np.append(Pool_Train_Loss, Train_Loss, axis=1)
		Pool_Valid_Acc = np.append(Pool_Valid_Acc, Valid_Acc, axis=1)
		Pool_Train_Acc = np.append(Pool_Train_Acc, Train_Acc, axis=1)	

		print('Evaluate Model Test Accuracy with pooled points')

		score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
		print('Test score:', score)
		print('Test accuracy:', acc)
		all_accuracy = np.append(all_accuracy, acc)


		print('Use this trained model with pooled points for Dropout again')

	print('Storing Accuracy Values over experiments')
	Experiments_All_Accuracy = Experiments_All_Accuracy + all_accuracy


	print('Saving Results Per Experiment')
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Paper_Submission/Binary_Classification/Results/'+'Random_Q10_N1000_Train_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Loss)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Paper_Submission/Binary_Classification/Results/'+ 'Random_Q10_N1000_Valid_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Loss)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Paper_Submission/Binary_Classification/Results/'+'Random_Q10_N1000_Train_Acc_'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Acc)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Paper_Submission/Binary_Classification/Results/'+ 'Random_Q10_N1000_Valid_Acc_'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Acc)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Paper_Submission/Binary_Classification/Results/'+'Random_Q10_N1000_Pooled_Image_Index_'+ 'Experiment_' + str(e) + '.npy', x_pool_All)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Paper_Submission/Binary_Classification/Results/'+ 'Random_Q10_N1000_Accuracy_Results_'+ 'Experiment_' + str(e) + '.npy', all_accuracy)

print('Saving Average Accuracy Over Experiments')

Average_Accuracy = np.divide(Experiments_All_Accuracy, Experiments)

np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Paper_Submission/Binary_Classification/Results/'+'Random_Q10_N1000_Average_Accuracy'+'.npy', Average_Accuracy)












