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

X_train = X_train[0:10000, 0:3,0:32,0:32]
y_train = y_train[0:10000, 0]

X_test = X_test[0:1000,0:3,0:32,0:32]
y_test = y_test[0:1000,0]


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
# predict - generates output predictions - returns numpy array of predictions - returns raw predictions?

#we chaged the code - to make PREDICT the same as PREDICT_STOCHASTIC
score = model.predict(X_test, batch_size=batch_size)
#score = model.predict_stochastic(X_test,batch_size=batch_size)

print('PRINTING PREDICT RESULTS')
print('Test score:', score)
np.savetxt("Predict_Results.csv", score, delimiter=",")


#evaluate - returns the scalar test loss - computes the loss on some input data
evaluation_score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('PRINTING EVALUATE RESULTS')
print('Test score:', evaluation_score[0])
print('Test accuracy:', evaluation_score[1])

np.savetxt("Evaluate_Results.csv", score, delimiter=",")




#predict_classes - would generate class predictions - numpy array of class predictions
score_classes = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
print('PRINTING SCORE CLASSES RESULTS')
print('Score Classes', score_classes)
np.savetxt("Predict_Classes.csv", score_classes, delimiter=",")



#predict_proba = generates class probability predictions 
score_proba = model.predict_proba(X_test, batch_size=batch_size, verbose=1)
print('PRINTING SCORE PROBABILITIES')
print('Score Probabilities', score_proba)
np.savetxt("Score_Probabilities.csv", score_proba, delimiter=",")


# SCORE PROBABILITIES SAME AS PREDICT ---- PREDICT_RESULTS.CSV file