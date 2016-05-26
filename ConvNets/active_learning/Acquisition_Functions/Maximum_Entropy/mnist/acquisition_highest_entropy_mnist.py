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
nb_epoch = 10

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


X_valid = X_train_All[2000:3500, :, :, :]
y_valid = y_train_All[2000:3500]

X_train = X_train_All[0:1875, :, :, :]
y_train = y_train_All[0:1875]

X_Pool = X_train_All[4000:44000, :, :, :]
y_Pool = y_train_All[4000:44000]

#using the entire test set - 10,000 points
# X_test = X_test[0:200, :, :, :]
# y_test = y_test[0:200]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

score=0
all_accuracy = 0
acquisition_iterations = 10

#Number of Queries to make every iteration
Queries = 100

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


#this takes the loss after several epochs - when model converged - and checks loss after every pooling
# Pool_Train_Loss = 					# loss after every pooling
# Pool_Test_Loss = 


Pool_Valid_Loss = np.zeros(shape=(nb_epoch, 1)) 	#row - no.of epochs, col (gets appended) - no of pooling
Pool_Train_Loss = np.zeros(shape=(nb_epoch, 1)) 
x_pool_All = np.zeros(shape=(1))


# i denotes the number of times to concatenate or perform acquisition from the pool data
for i in range(acquisition_iterations):

	print('POOLING ITERATION NUMBER', i)

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

	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
	Train_Result_Optimizer = hist.history

	# loss vs Number of Epochs - to check model overfitting and convergence
	Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	Train_Loss = np.array([Train_Loss]).T
	Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
	Valid_Loss = np.asarray([Valid_Loss]).T


	#Accumulate the training and validation/test loss after every pooling iteration - for plotting
	Pool_Valid_Loss = np.append(Pool_Valid_Loss, Valid_Loss, axis=1)
	Pool_Train_Loss = np.append(Pool_Train_Loss, Train_Loss, axis=1)		


	Class_Probability = model.predict_proba(X_Pool, batch_size=batch_size, verbose=1)
	Class_Log_Probability = np.log2(Class_Probability)
	Entropy_Each_Cell = - np.multiply(Class_Probability, Class_Log_Probability)

	Entropy = np.sum(Entropy_Each_Cell, axis=1)	# summing across rows of the array

	#x_pool_index = 	np.unravel_index(Entropy.argmax(), Entropy.shape)	#for finding the maximum value np.amax(Entropy)
	x_pool_index = Entropy.argsort()[-Queries:][::-1]

	# THIS FINDS THE INDEX OF THE MINIMUM
	# a_1d = Entropy.flatten()
	# x_pool_index = a_1d.argsort()[-N:]

		#saving pooled images
	for im in range(x_pool_index.shape[0]):
		Image = X_Pool[x_pool_index[im], :, :, :]
		img = Image.reshape((28,28))
		sp.misc.imsave('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Maximum_Entropy/mnist/Pooled_Images/'+'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)

	#store all the pooled images indexes
	x_pool_All = np.append(x_pool_All, x_pool_index)


	Pooled_X = X_Pool[x_pool_index, :, :, :]
	Pooled_Y = y_Pool[x_pool_index]		# true label from the oracle

	X_train = np.concatenate((X_train, Pooled_X), axis=0)
	y_train = np.concatenate((y_train, Pooled_Y), axis=0)

	#delete the currently pooled points from the pool set
	X_Pool = np.delete(X_Pool, x_pool_index, 0)

	print('After random acquisitions')
	print(X_train.shape[0], 'train samples after acquisition')
	print(X_test.shape[0], 'test samples')
	print(X_valid.shape[0], 'validation samples')
	print(X_Pool.shape[0], 'Pooled X samples')


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

np.savetxt("Highest Entropy Accuracy Values.csv", all_accuracy, delimiter=",")



np.save('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Maximum_Entropy/mnist/Results/'+'All_Train_Loss'+'.npy', Pool_Train_Loss)
np.save('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Maximum_Entropy/mnist/Results/'+ 'All_Valid_Loss'+'.npy', Pool_Valid_Loss)
np.save('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Maximum_Entropy/mnist/Results/'+'All_Pooled_Image_Index'+'.npy', x_pool_All)
np.save('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Maximum_Entropy/mnist/Results/'+ 'All_Accuracy_Results'+'.npy', all_accuracy)



#to load again, and visualize the plots:
#score4 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/BCNN_Maximal_Uncertainty/Bayes_Segnet/Dropout_Scores/'+'Dropout_Score_4.npy')


# plt.figure(figsize=(8, 6), dpi=80)
# plt.clf()
# plt.hold(1)
# plt.plot(Pool_Train_Loss, color="blue", linewidth=1.0, marker='o', label="Training categorical_crossentropy loss")
# plt.plot(Pool_Valid_Loss, color="red", linewidth=1.0, marker='o', label="Validation categorical_crossentropy loss")
# plt.xlabel('Number of Epochs')
# plt.ylabel('Categorical Cross Entropy Loss Function')
# plt.title('Training and Validation Set Loss Function and Convergence')
# plt.grid()
# plt.xlim(0, nb_epoch)
# plt.ylim(0, 0.5)
# plt.legend(loc = 4)
# plt.show()



All_Accuracy = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Maximum_Entropy/mnist/All_Accuracy.npy')
Pool_Train_Loss = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Maximum_Entropy/mnist/Pool_Train_Loss.npy')
Pool_Valid_Loss = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/active_learning/Acquisition_Functions/Maximum_Entropy/mnist/Pool_Valid_Loss.npy')



'''

plt.plot(Pool_Train_Loss_Plot[:,3], color="blue", label="Original Data")
plt.plot(Pool_Train_Loss_Plot[:,0], color="red", label="After 4 Pooling")
plt.plot(Pool_Valid_Loss_Plot[:,3], color="black", label="Original Data")
plt.plot(Pool_Valid_Loss_Plot[:,0], color="green", label="After 4 Pooling")

'''

