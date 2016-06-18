#minimum expected entropy
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

Experiments = 2

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
acquisition_iterations = 29

Queries = 100


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


	X_valid = X_train_All[10000:11000, :, :, :]
	y_valid = y_train_All[10000:11000]

	X_Pool = X_train_All[20000:60000, :, :, :]
	y_Pool = y_train_All[20000:60000]


	X_train_All = X_train_All[0:10000, :, :, :]
	y_train_All = y_train_All[0:10000]


	#training data to have equal distribution of classes
	idx_0 = np.array( np.where(y_train_All==0)  ).T
	idx_0 = idx_0[0:10,0]
	X_0 = X_train_All[idx_0, :, :, :]
	y_0 = y_train_All[idx_0]

	idx_1 = np.array( np.where(y_train_All==1)  ).T
	idx_1 = idx_1[0:10,0]
	X_1 = X_train_All[idx_1, :, :, :]
	y_1 = y_train_All[idx_1]

	idx_2 = np.array( np.where(y_train_All==2)  ).T
	idx_2 = idx_2[0:10,0]
	X_2 = X_train_All[idx_2, :, :, :]
	y_2 = y_train_All[idx_2]

	idx_3 = np.array( np.where(y_train_All==3)  ).T
	idx_3 = idx_3[0:10,0]
	X_3 = X_train_All[idx_3, :, :, :]
	y_3 = y_train_All[idx_3]

	idx_4 = np.array( np.where(y_train_All==4)  ).T
	idx_4 = idx_4[0:10,0]
	X_4 = X_train_All[idx_4, :, :, :]
	y_4 = y_train_All[idx_4]

	idx_5 = np.array( np.where(y_train_All==5)  ).T
	idx_5 = idx_5[0:10,0]
	X_5 = X_train_All[idx_5, :, :, :]
	y_5 = y_train_All[idx_5]

	idx_6 = np.array( np.where(y_train_All==6)  ).T
	idx_6 = idx_6[0:10,0]
	X_6 = X_train_All[idx_6, :, :, :]
	y_6 = y_train_All[idx_6]

	idx_7 = np.array( np.where(y_train_All==7)  ).T
	idx_7 = idx_7[0:10,0]
	X_7 = X_train_All[idx_7, :, :, :]
	y_7 = y_train_All[idx_7]

	idx_8 = np.array( np.where(y_train_All==8)  ).T
	idx_8 = idx_8[0:10,0]
	X_8 = X_train_All[idx_8, :, :, :]
	y_8 = y_train_All[idx_8]

	idx_9 = np.array( np.where(y_train_All==9)  ).T
	idx_9 = idx_9[0:10,0]
	X_9 = X_train_All[idx_9, :, :, :]
	y_9 = y_train_All[idx_9]

	X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
	y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )


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
	Y_train = np_utils.to_categorical(y_train, nb_classes)

	#loss values in each experiment
	Pool_Valid_Loss = np.zeros(shape=(nb_epoch, 1)) 	
	Pool_Train_Loss = np.zeros(shape=(nb_epoch, 1)) 
	Pool_Valid_Acc = np.zeros(shape=(nb_epoch, 1)) 	
	Pool_Train_Acc = np.zeros(shape=(nb_epoch, 1)) 
	x_pool_All = np.zeros(shape=(1))

	
	print('Training Model Without Acquisitions in Experiment', e)

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
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=(X_valid, Y_valid))
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

	print('Evaluating Test Accuracy Without Acquisition')
	score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

	all_accuracy = acc

	print('Starting Active Learning in Experiment ', e)

	for i in range(acquisition_iterations):
		print('POOLING ITERATION', i)

		Y_train = np_utils.to_categorical(y_train, nb_classes)
		Y_test = np_utils.to_categorical(y_test, nb_classes)

		All_X_Pool = X_Pool
		All_Y_Pool = y_Pool

		#take a random subset of the pool points - 500 for computational efficiency - only 10 queries will be selected from 500
		pool_subset = 500
		pool_subset_dropout = np.asarray(random.sample(range(0,X_Pool.shape[0]), pool_subset))
		X_Pool = All_X_Pool[pool_subset_dropout, :, :, :]
		y_Pool = All_Y_Pool[pool_subset_dropout]

		# X_Pool.shape[0]			# for each unlabelled image
		for pool_point in range(X_Pool.shape[0]):
			for pool_label in range(nb_classes):

				#estimate P(Y_i =j | L)
				# L - labelled points
				Labelled_data_X = X_train
				Labelled_data_Y = Y_train

				model = Sequential()
				model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
				model.add(Activation('relu'))   #using relu activation function
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

				model.fit(Labelled_data_X, Labelled_data_Y, batch_size=batch_size, nb_epoch=nb_epoch)

				print('Estimating P(Y=j | Labelled Samples)')
				Class_Probability = model.predict_proba(X_Pool, batch_size=batch_size, verbose=0)

				index = np.array([pool_point])
				i_pool_x = X_Pool[index, :, :, :]
				i_pool_y = Y_Pool[index, :]

				#all other pool points other than current point
				Remaining_Pool_Points = np.delete(X_Pool, pool_point, 0)

				for rm_pool_point in range(Remaining_Pool_Points.shape[0]):
					for pool_class in range(nb_classes):

						New_Labelled_data_X = np.concatenate((Labelled_data_X, i_pool_x), axis=0)
						New_Labelled_data_Y = np.concatenate((Labelled_data_Y, i_pool_y), axis=0)

						model.fit(New_Labelled_data_X, New_Labelled_data_Y, batch_size=batch_size, nb_epoch=nb_epoch)

						print('Estimate P(Y=l | L U {x, j})')
						# set of P(Y_k = l | L U (x_i, j)) values0
						P_Y_k = model.predict_proba(Remaining_Pool_Points, batch_size=batch_size, verbose=0)

						log_P_Y_k = np.log2(P_Y_k)
						Entropy_Y_k = - np.multiply(P_Y_k, log_P_Y_k)

						#Entropy_Y_k = np.sum(Entropy_Y_k_Each_Cell, axis=1)	# summing across rows of the array

			print('Computing the expected entropy')	
			H_x = np.array([[     np.dot(   Class_Probability[pool_point , :],  Entropy_Y_k[pool_point, :] )     ]])
			H_x = np.append(H_x, H_x, axis=0)


		print('Choosing pool point that results in minimum expected entropy')
		# acquise the point with argmin (H_x) and its index value

		#THIS FINDS THE MINIMUM INDEX 
		a_1d = H_x.flatten()
		acquised_point_index = a_1d.argsort()[-Queries:]


		#for choosing only 1 point
		# acquised_point = np.amin(H_x)
		# acquised_point_index = np.where(H_x==H_x.min())[0]
		#acquised_point_index = H_x.argsort()[:1]

		Pooled_X = X_Pool[acquised_point_index, :, :, :]
		Pooled_Y = y_Pool[acquised_point_index]

		# accumulate all the acquised points from X_Pool - accumulate all indices of the acquised points
		# concatenate all the acquised points with the training data
		X_train = np.concatenate((X_train, Pooled_X), axis=0)
		y_train = np.concatenate((y_train, Pooled_Y), axis=0)


		print('Test Model Accuracy with new acquised point')
		# Compute the test error and accuracy 
		score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
		print('Test score:', score)
		print('Test accuracy:', acc)
		all_accuracy = np.append(all_accuracy, acc)


	print('Storing Accuracy Values over experiments')
	Experiments_All_Accuracy = Experiments_All_Accuracy + all_accuracy

	print('Saving Results Per Experiment')
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Expected_Entropy/Results/'+'Averaged_Main_MEE_Q100_N3000_Train_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Loss)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Expected_Entropy/Results/'+ 'Averaged_Main_MEE_Q100_N3000_Valid_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Loss)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Expected_Entropy/Results/'+'Averaged_Main_MEE_Q100_N3000_Train_Acc_'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Acc)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Expected_Entropy/Results/'+ 'Averaged_Main_MEE_Q100_N3000_Valid_Acc_'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Acc)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Expected_Entropy/Results/'+'Averaged_Main_MEE_Q100_N3000_Pooled_Image_Index_'+ 'Experiment_' + str(e) + '.npy', x_pool_All)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Expected_Entropy/Results/'+ 'Averaged_Main_MEE_Q100_N3000_Accuracy_Results_'+ 'Experiment_' + str(e) + '.npy', all_accuracy)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Expected_Entropy/Results/'+ 'Averaged_Main_MEE_Q100_N3000_rmse_Results_'+ 'Experiment_' + str(e) + '.npy', all_rmse)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Expected_Entropy/Results/'+ 'Averaged_Main_MEE_Q100_N3000_logLikelihood_Results_'+ 'Experiment_' + str(e) + '.npy', all_predicted_log_likelihood)


print('Saving Average Accuracy Over Experiments')

Average_Accuracy = np.divide(Experiments_All_Accuracy, Experiments)

np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Expected_Entropy/Results/'+'Averaged_Main_MEE_Q100_N3000_Average_Accuracy'+'.npy', Average_Accuracy)















			
			
			

		




