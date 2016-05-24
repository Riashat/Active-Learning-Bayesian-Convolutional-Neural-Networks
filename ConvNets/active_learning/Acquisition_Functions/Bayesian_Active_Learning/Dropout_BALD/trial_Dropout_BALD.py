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


X_valid = X_train_All[2000:3000, :, :, :]
y_valid = y_train_All[2000:3000]

X_train = X_train_All[0:2000, :, :, :]
y_train = y_train_All[0:2000]

X_Pool = X_train_All[3000:20000, :, :, :]
y_Pool = y_train_All[3000:20000]

X_test = X_test[0:2000, :, :, :]
y_test = y_test[0:2000]

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

score=0
all_accuracy = 0
acquisition_iterations = 1
dropout_iterations = 5
Queries = 100   # number of aquisitions per iteration




for i in range(acquisition_iterations):
	print('POOLING ITERATION', i)
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


	for d in range(dropout_iterations):
		print ('Dropout Iteration', d)
		score = model.predict(X_Pool,batch_size=batch_size, verbose=1)
		np.save('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Bayesian_Active_Learning/Dropout_BALD/Dropout_Scores/'+'Dropout_Score_'+str(d)+'.npy',score)


	score0 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Bayesian_Active_Learning/Dropout_BALD/Dropout_Scores/'+'Dropout_Score_0.npy')
	score1 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Bayesian_Active_Learning/Dropout_BALD/Dropout_Scores/'+'Dropout_Score_1.npy')
	score2 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Bayesian_Active_Learning/Dropout_BALD/Dropout_Scores/'+'Dropout_Score_2.npy')
	score3 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Bayesian_Active_Learning/Dropout_BALD/Dropout_Scores/'+'Dropout_Score_3.npy')
	score4 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Bayesian_Active_Learning/Dropout_BALD/Dropout_Scores/'+'Dropout_Score_4.npy')


	All_Pi = score0 + score1 + score2 + score3 + score4
	Avg_Pi = np.divide(All_Pi, dropout_iterations)
	Log_Avg_Pi = np.log2(Avg_Pi)
	Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
	Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

	G_X = Entropy_Average_Pi

	H_Matrix = np.zeros(shape=(score.shape[0], dropout_iterations))


	Log_Score0 = np.log2(score0)
	Entropy0 = - np.multiply(score0, Log_Score0)
	Entropy_Score0 = np.sum(Entropy0, axis=1)

	Log_Score1 = np.log2(score1)
	Entropy1 = - np.multiply(score1, Log_Score1)
	Entropy_Score1 = np.sum(Entropy1, axis=1)

	Log_Score2 = np.log2(score2)
	Entropy2 = - np.multiply(score2, Log_Score1)
	Entropy_Score2 = np.sum(Entropy2, axis=1)
	
	Log_Score3 = np.log2(score3)
	Entropy3 = - np.multiply(score1, Log_Score3)
	Entropy_Score3 = np.sum(Entropy3, axis=1)
	
	Log_Score4 = np.log2(score4)
	Entropy4 = - np.multiply(score4, Log_Score4)
	Entropy_Score4 = np.sum(Entropy4, axis=1)
	

	All_Entropy = Entropy_Score0 + Entropy_Score1 + Entropy_Score2 + Entropy_Score3 + Entropy_Score4 
	Average_Entropy = np.divide(All_Entropy, dropout_iterations)

	F_X = Average_Entropy

	U_X = G_X - F_X


	# THIS FINDS THE MINIMUM INDEX 
	# a_1d = U_X.flatten()
	# x_pool_index = a_1d.argsort()[-Queries:]

	a_1d = U_X.flatten()
	x_pool_index = a_1d.argsort()[-Queries:][::-1]



	#saving pooled images
	for im in range(x_pool_index.shape[0]):
		Image = X_Pool[x_pool_index[im], :, :, :]
		img = Image.reshape((28,28))
		sp.misc.imsave('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Bayesian_Active_Learning/Dropout_BALD/Pooled_Images/'+'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)


	Pooled_X = X_Pool[x_pool_index, 0:3,0:32,0:32]
	Pooled_Y = y_Pool[x_pool_index]	


	delete_Pool_X = np.delete(X_Pool, (x_pool_index), axis=0)
	delete_Pool_Y = np.delete(y_Pool, (x_pool_index), axis=0)

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

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))
	# Train_Result_Optimizer = hist.history
	# Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	# Train_Loss = np.array([Train_Loss]).T
	# Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
	# Valid_Loss = np.asarray([Valid_Loss]).T

	print('EVALUATING THE MODEL ON TEST SET')

	print('Evaluating Test Accuracy')
	score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
	print('Test score:', score)
	print('Test accuracy:', acc)
	all_accuracy = np.append(all_accuracy, acc)




# print('SIZE OF TRAINING DATA AFTER ACQUISITIONS', X_train.shape)

# print('TEST THE MODEL ACCURACY')
# # Compute the test error and accuracy 
# score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
# print('Test score:', score)
# print('Test accuracy:', acc)

# all_accuracy = np.append(all_accuracy, acc)

np.savetxt("Dropout BALD Accuracy Values.csv", all_accuracy, delimiter=",")


# plt.figure(figsize=(8, 6), dpi=80)
# plt.clf()
# plt.hold(1)
# plt.plot(Train_Loss, color="blue", linewidth=1.0, marker='o', label="Training categorical_crossentropy loss")
# plt.plot(Valid_Loss, color="red", linewidth=1.0, marker='o', label="Validation categorical_crossentropy loss")
# plt.xlabel('Number of Epochs')
# plt.ylabel('Categorical Cross Entropy Loss Function')
# plt.title('Training and Validation Set Loss Function and Convergence')
# plt.grid()
# plt.xlim(0, nb_epoch)
# plt.ylim(0, 0.5)
# plt.legend(loc = 4)
# plt.show()

