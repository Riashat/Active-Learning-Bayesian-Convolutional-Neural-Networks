'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

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
nb_epoch = 200
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols),name='conv1_1'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3,name='conv1_2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same',name='conv2_1'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3,name='conv2_2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,name='dense_1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes,name='dense_2'))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = 'dense_2'
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer
# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].get_output()
print("YODA Layer-output shape",layer_output.shape)
loss = K.mean(layer_output[:,filter_index])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)



input_img_data = [X_train[0,:,:,:]]
sp.misc.imsave('test.jpg',input_img_data)



input_img = model.get_input() 
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
grads = K.gradients(loss,input_img)
iterate = K.function([input_img], [loss, grads])


print("YODA_1")
step = 0.01
for i in range(10):
   loss_value, grads_value = iterate([input_img_data])
   input_img_data += grads_value*step
score = model.predict_stochastic(input_img_data,batch_size=batch_size)
print(score)
print("YODA")
json_string = model.to_json()
open('model_200_arch.json', 'w').write(json_string)
model.save_weights('model_200_weights.h5')


for i in range(1):
   score = model.predict_stochastic(X_test,batch_size=batch_size)
   np.save('/home/ar773/CNN/keras/examples/scores/score'+str(i)+'.npy',score)
print('Test score:', score)
loop the predict 10 times, average over the trials and take a argmax for correct label
