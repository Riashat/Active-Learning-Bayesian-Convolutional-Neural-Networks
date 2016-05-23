from __future__ import print_function
from keras.datasets import mnist
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


batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# X_Pool = X_train[10000:60000, 0:1, 0:28, 0:28]
# Y_Oracle = Y_train[10000:60000, :]

# X_train = X_train[0:10000, 0:1, 0:28, 0:28]
# Y_train = Y_train[0:10000, :]

X_train_Seg = X_train[0:12000, :, :, :]
Y_train_Seg = Y_train[0:12000, :]
X_test = X_test[0:2000, :, :, :]
Y_test = Y_test[0:2000, :]

X_Pool = X_train[4000:60000, :, :, :]
Y_Oracle = Y_train[4000:60000, :] 

X_train = X_train_Seg[0:4000, :, :, :]
Y_train = Y_train_Seg[0:4000, :]

X_valid = X_train_Seg[10000:12000, :, :, :]
Y_valid = Y_train_Seg[10000:12000, :]

score=0
loss = 0
data_size = X_train.shape[0]

#load uncertainty results
uncertainty = scipy.io.loadmat('/Users/Riashat/Documents/Cambridge_THESIS/Experiments/keras/active_learning/Acquisition_Functions/BCNN_Maximal_Uncertainty/mnist/Uncertainty_Results.mat')
All_Mean = uncertainty['All_Mean']
All_Std = uncertainty['All_Std']
Label_Prob = uncertainty['Label_Prob']
Std = uncertainty['Std']
Label_Class = uncertainty['Label_Class']
Y_pred = uncertainty['Y_pred']
BayesSegnet_Sigma = uncertainty['BayesSegnet_Sigma']

#all of the above are numpy arrays 

Pooled_Y = np.zeros(1)
Queries = 1000

score=0
accuracy=0
sklearn_accuracy=0
iterations = 1

All_Train_Loss = np.zeros(shape=(nb_epoch, 1))


for i in range(iterations):

	print('POOLING ITERATION', i)

	a_1d = BayesSegnet_Sigma.flatten()
	row = a_1d.argsort()[-Queries:]

	Pooled_X = X_Pool[row, 0:1, 0:28, 0:28]
	Pooled_Y = Y_Oracle[row,:]

	delete_std = np.delete(Std, (row), axis=0)
	delete_Pool_X = np.delete(X_Pool, (row), axis=0)

	X_train = np.concatenate((X_train, Pooled_X), axis=0)
	Y_train = np.concatenate((Y_train, Pooled_Y), axis=0)

	Y_true = np.zeros([Y_test.shape[0], 1])

	for y in range(Y_test.shape[0]):
		labs = Y_test[y,:]
		c = np.argmax(labs, axis=0)
		Y_true[y,:] = c 


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

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255

	model.compile(loss='categorical_crossentropy', optimizer='adadelta')
	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))
	Train_Result_Optimizer = hist.history
	Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	Train_Loss = np.array([Train_Loss]).T
	All_Train_Loss = np.concatenate((All_Train_Loss, Train_Loss), axis=1)

	Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
	Valid_Loss = np.asarray([Valid_Loss]).T




Y_pred = model.predict_classes(X_test, batch_size=batch_size, verbose=1)

# using model.predict to measure accuracy directly?
# acc = accuracy(y_test, np.round(np.array(model.predict({'input': X_test}, batch_size=batch_size)['output'])))

#### THIS Y_PRED SHOULD BE FROM THIS MODEL
print('Using SKLEARN ACCURACY MEASURE')
sk_accuracy = accuracy_score(Y_true, Y_pred)

print('sk_accuracy', sk_accuracy)

sklearn_accuracy = np.append(sklearn_accuracy, sk_accuracy)

print('sklearn_accuracy', sklearn_accuracy)


np.savetxt("Acquisition_Accuracy.csv", sklearn_accuracy, delimiter=",")


plt.figure(figsize=(8, 6), dpi=80)
plt.clf()
plt.hold(1)
plt.plot(Train_Loss, color="blue", linewidth=1.0, marker='o', label="Training categorical_crossentropy loss")
plt.plot(Valid_Loss, color="red", linewidth=1.0, marker='o', label="Validation categorical_crossentropy loss")
plt.xlabel('Number of Epochs')
plt.ylabel('Categorical Cross Entropy Loss Function')
plt.title('Training and Validation Set Loss Function and Convergence')
plt.grid()
plt.xlim(0, nb_epoch)
plt.ylim(0, 0.5)
plt.legend(loc = 4)
plt.show()


