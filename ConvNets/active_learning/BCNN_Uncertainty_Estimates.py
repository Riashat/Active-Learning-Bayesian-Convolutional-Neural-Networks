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


batch_size = 32
nb_classes = 10
nb_epoch = 1
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('Original size of the cifar10 dataset')
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

print ('Using poritions of the training and test sets')

# X_train = X_train[0:1000, 0:3,0:32,0:32]
# y_train = y_train[0:1000, 0]

X_train = X_train[0:100, 0:3,0:32,0:32]
y_train = y_train[0:100, 0]



X_test = X_test[0:100,0:3,0:32,0:32]
y_test = y_test[0:100,0]


print(X_test.shape[0])

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
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

#Flatten used to Flatten the input : 64*32*32 to the product of these (65536)

# Dense: just the regular fully connected NN layer
# Dense(512) means - output arrays of shape (*, 512)
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))



# let's train the model using SGD + momentum (how original).
# SGD from Optimizers
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Once your model looks good, configure its learning process with .compile():
# also using the Objective function "categorical_crossentropy" or can use "mean_squared_error" here
model.compile(loss='categorical_crossentropy', optimizer=sgd)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('Shape after this /= 255')
print('X_train shape:', X_train.shape)
print('X_test shape:' , X_test.shape)

print('TRAINING THE MODEL')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)

print('******* DONE WITH TRAINING ********')

print('******* TESTING THE MODEL ********')

score = model.predict_proba(X_test,batch_size=batch_size, verbose=1)		#score is a numpy array


scipy.io.savemat('Score5.mat', dict(score5 = score))







# Finding the mean of the 10 predictions

# All_Mean = np.zeros((X_test.shape[0], 10))
# All_Std = np.zeros((X_test.shape[0], 10))


# for i in range(10):
# 	score = model.predict_proba(X_test,batch_size=batch_size, verbose=1)		#score is a numpy array
# 	np.save('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores'+str(i), score)


# scores0 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores0.npy')
# scores1 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores1.npy')
# scores2 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores2.npy')
# scores3 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores3.npy')
# scores4 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores4.npy')
# scores5 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores5.npy')
# scores6 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores6.npy')
# scores7 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores7.npy')
# scores8 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores8.npy')
# scores9 = np.load('/Users/Riashat/Documents/Cambridge_THESIS/Project_Work/CNNs/keras/active_learning/scores9.npy')



# for j in range(scores1.shape[0]):
# 	for k in range(scores1.shape[1]):
# 		All_Mean[j,k] = (scores0[j,k] + scores1[j,k] + scores2[j,k] + scores3[j,k] + scores4[j,k] + scores5[j,k] + scores6[j,k] + scores7[j,k] + scores8[j,k] + scores9[j,k] )/10
# 		#List = np.array([[ scores0[j,k], scores1[j,k], scores2[j,k], scores3[j,k], scores4[j,k], scores5[j,k], scores6[j,k], scores7[j,k], scores8[j,k], scores9[j,k] ]])   





#Finding the standard deviation









