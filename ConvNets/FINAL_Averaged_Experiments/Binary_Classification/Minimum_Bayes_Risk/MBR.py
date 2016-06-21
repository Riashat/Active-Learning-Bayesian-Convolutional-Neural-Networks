#Minimum Bayes Risk
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
from scipy.spatial.distance import pdist, squareform
from scipy import linalg

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
acquisition_iterations = 2

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

	X_train = X_train_All[0:100, :, :, :]
	y_train = y_train_All[0:100]


	#use 10000 pool points to start with
	X_Pool = X_train_All[5000:10000, :, :, :]
	y_Pool = y_train_All[5000:10000]
	
	#we can train the model and evaluate the test accuracy with original size of training data here
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

	Pool_Valid_Loss = np.zeros(shape=(nb_epoch, 1)) 	#row - no.of epochs, col (gets appended) - no of pooling
	Pool_Train_Loss = np.zeros(shape=(nb_epoch, 1)) 
	Pool_Valid_Acc = np.zeros(shape=(nb_epoch, 1)) 	
	Pool_Train_Acc = np.zeros(shape=(nb_epoch, 1)) 

	print('Training Model Without Acquisitions in Experiment', e)


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

	print('Evaluating Test Accuracy Without Acquisition')
	score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

	all_accuracy = acc

	print('Performing Active Learning')

	for i in range(acquisition_iterations):

		print('POOLING ITERATION ', i)

		All_Data = np.concatenate( (X_train, X_Pool), axis=0  )

		Image_Data = All_Data.reshape(All_Data.shape[0], img_rows*img_cols)

		#compute the kernel matrix
		sigma  = 1
		pairwise_dists = squareform(pdist(Image_Data, 'euclidean'))
		W = sp.exp(pairwise_dists ** 2 / sigma ** 2)


		#compute the combinatorial Laplacian
		d_i = W.sum(axis=1)
		D = np.diag(d_i)

		Delta = D - W 

		#computing the harmonic function - without any acquisitions yet
		Delta_ll = Delta[0:X_train.shape[0], 0:X_train.shape[0]]
		Delta_ul = Delta[X_train.shape[0]:, 0:X_train.shape[0]]
		Delta_lu = Delta[0:X_train.shape[0], X_train.shape[0]:]
		Delta_uu = Delta[X_train.shape[0]:, X_train.shape[0]:]


		inv_Delta_uu = linalg.inv(Delta_uu)
		Original_f_L = y_train
		Delta_mult = np.dot(inv_Delta_uu, Delta_ul)
		Original_f_U = - np.dot(Delta_mult, Original_f_L)			

		#f_I is the entire harmonic function over all the data points (U + L)
		Original_f_I = np.concatenate((Original_f_L, Original_f_U), axis=0)

		print('Compute Expected Bayes Risk for ALL Pool Points in Acquisition Iteration ', i)

		Bayes_Risk = np.zeros(shape=(X_Pool.shape[0]))

		#use a subset of the pool points - for Q=10, use 500 pool point subset
		Pool_Subset = 50


		for k in range(Pool_Subset):

			print('Pool Subset Iteration', k)

			#compute estimated risk for each added point
			Pool_Point = X_Pool[np.array([k]), :, :, :]
			Pool_Point_y = y_Pool[np.array([k])]

			#add this pool point to labelled data - but we don't know the actual label for it
			X_train_Temp = np.concatenate((X_train, Pool_Point), axis=0)
			y_train_Temp = np.concatenate((y_train, Pool_Point_y), axis=0)

			#delete this pool point from pool set
			X_Pool_Temp = np.delete(X_Pool, k, 0)


			# # W and D stays the same - only Delta_uu, Delta_ul etc changes
			Delta_ll = Delta[0:X_train_Temp.shape[0], 0:X_train_Temp.shape[0]]
			Delta_ul = Delta[X_train_Temp.shape[0]:, 0:X_train_Temp.shape[0]]
			Delta_lu = Delta[0:X_train_Temp.shape[0], X_train_Temp.shape[0]:]
			Delta_uu = Delta[X_train_Temp.shape[0]:, X_train_Temp.shape[0]:]


			#compute the new changed f
			inv_Delta_uu = linalg.inv(Delta_uu)
			f_L = y_train_Temp
			Delta_mult = np.dot(inv_Delta_uu, Delta_ul)
			f_U = - np.dot(Delta_mult, f_L)			
			f_I = np.concatenate((f_L, f_U), axis=0)

			#compute the new estimated Bayes risk for this added point
			R = np.array([0])
			for m in range(f_I.shape[0]):
				val_f_I = f_I[m]
				other_val_f_I = 1 - val_f_I
				min_val = np.amin(np.array([val_f_I, other_val_f_I]))
				R = R + min_val
			Estimated_Risk = R

			#we need f_k values for each pool point in consideration
			f_All_Pool = Original_f_I[Original_f_L.shape[0]:]
			f_k = f_All_Pool[k]

			Bayes_Risk[k] = (1 - f_k) * Estimated_Risk + (f_k)*Estimated_Risk

		print('Finished Computing Bayes Risk for Unlabelled Pool Points')

		#find the best query from the Bayes_Risk - do the acquisition	
		# THIS FINDS THE INDEX OF THE MINIMUM
		b_1d = Bayes_Risk.flatten()
		x_pool_index = b_1d.argsort()[-Queries:]

		#find the query point x_pool_index from Pool Set and its original label
		Pooled_X = X_Pool[x_pool_index, :, :, :]
		Pooled_Y = y_Pool[x_pool_index]		# true label from the oracle

		#add queried point to train set and remove from pool set
		print('Acquised Points added to training set')
		X_train = np.concatenate((X_train, Pooled_X), axis=0)
		y_train = np.concatenate((y_train, Pooled_Y), axis=0)

		#delete the currently pooled points from the pool set
		X_Pool = np.delete(X_Pool, x_pool_index, 0)


		print('Training Model with pooled points')

		# convert class vectors to binary class matrices
		Y_train = np_utils.to_categorical(y_train, nb_classes)

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
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Bayes_Risk/Results/'+'Averaged_Main_MBR_Q100_N3000_Train_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Loss)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Bayes_Risk/Results/'+ 'Averaged_Main_MBR_Q100_N3000_Valid_Loss_'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Loss)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Bayes_Risk/Results/'+'Averaged_Main_MBR_Q100_N3000_Train_Acc_'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Acc)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Bayes_Risk/Results/'+ 'Averaged_Main_MBR_Q100_N3000_Valid_Acc_'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Acc)
	np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Bayes_Risk/Results/'+ 'Averaged_Main_MBR_Q100_N3000_Accuracy_Results_'+ 'Experiment_' + str(e) + '.npy', all_accuracy)

print('Saving Average Accuracy Over Experiments')

Average_Accuracy = np.divide(Experiments_All_Accuracy, Experiments)

np.save('/home/ri258/Documents/Project/MPhil_Thesis_Cluster_Experiments/ConvNets/Cluster_Experiments/Minimum_Bayes_Risk/Results/'+'Averaged_Main_MBR_Q100_N3000_Average_Accuracy'+'.npy', Average_Accuracy)






