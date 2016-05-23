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

print('Original size of the cifar10 dataset')
print('X_train shape:', X_train_All.shape)
print('y_train shape:', y_train_All.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

train_points = 2000
test_points = 1000

# print ('Using poritions of the training and test sets')
X_train = X_train_All[0:train_points, 0:3,0:32,0:32]
y_train = y_train_All[0:train_points, :]

X_test = X_test[0:test_points,0:3,0:32,0:32]
y_test = y_test[0:test_points,:]
#pool of training data points
X_Pool = X_train_All[1001:50000, 0:3, 0:32, 0:32]
Y_oracle = y_train_All[1001:50000, :]

# no labels available for the Pool Training Points

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

score=0
loss = 0
data_size = X_train.shape[0]


#load uncertainty results
uncertainty = scipy.io.loadmat('/Users/Riashat/Documents/Cambridge_THESIS/Experiments/keras/active_learning/Acquisition_Functions/BCNN_Maximal_Uncertainty/cifar10/Uncertainty_Results.mat')
All_Mean = uncertainty['All_Mean']
All_Std = uncertainty['All_Std']
Label_Prob = uncertainty['Label_Prob']
Std = uncertainty['Std']
Label_Class = uncertainty['Label_Class']
Y_pred = uncertainty['Y_pred']

#all of the above are numpy arrays 

Pooled_Y = np.zeros(1)
Queries = 100

score=0
accuracy=0
sklearn_accuracy=0
iterations = 8

All_Train_Loss = np.zeros(shape=(nb_epoch, 1))

for i in range(iterations):

	print('POOLING ITERATION', i)

	a_1d = Std.flatten()
	row = a_1d.argsort()[-Queries:]

	Pooled_X = X_Pool[row, 0:3,0:32,0:32]
	Pooled_Y = Y_oracle[row,:]

	#DELETE that std value from
	# STD
	# DELETE PICKED POOL POINTS
	delete_std = np.delete(Std, (row), axis=0)
	delete_Pool_X = np.delete(X_Pool, (row), axis=0)

	X_train = np.concatenate((X_train, Pooled_X), axis=0)
	y_train = np.concatenate((y_train, Pooled_Y), axis=0)

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	Y_true = np.zeros([Y_test.shape[0], 1])


	for y in range(Y_test.shape[0]):
		labs = Y_test[y,:]
		c = np.argmax(labs, axis=0)
		Y_true[y,:] = c 


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
	X_test /= 255


	hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
	Train_Result_Optimizer = hist.history
	Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
	Train_Loss = np.array([Train_Loss]).T

	All_Train_Loss = np.concatenate((All_Train_Loss, Train_Loss), axis=1)



Y_train = np_utils.to_categorical(y_train, nb_classes)
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
X_test /= 255


model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)


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




# Having made N acquisitions, we evaluate the final model - train them with all the acquised pool points
# and then measure the accuracy on the test set
# plt.plot(All_Train_Loss)
# plt.ylabel('Training Loss with Acquisitions')
# plt.xlabel('Number of Epochs')
# plt.show()



