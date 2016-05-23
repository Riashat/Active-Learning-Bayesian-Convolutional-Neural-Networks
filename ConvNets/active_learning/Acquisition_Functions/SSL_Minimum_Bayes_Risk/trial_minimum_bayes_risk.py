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
import numpy as np
import math
from scipy import linalg


batch_size = 32
nb_classes = 10
nb_epoch = 1
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train_All, y_train_All), (X_test, y_test) = cifar10.load_data()

# print ('Using poritions of the training and test sets')
X_train = X_train_All[0:20000, 0:3,0:16,0:16]	#both labelled and unlabelled data
y_train = y_train_All[0:20000, :]

X_test = X_test[0:5000,0:3,0:16, 0:16]
y_test = y_test[0:5000,:]

# Labelled - first 500/20000
# Unlabelled - 500 - 20000
Labelled_Data_X = X_train[0:500,:, :, :]
Unlabelled_Data_X = X_train[500:2000, :, :, :]

#remove 500-2000 labels for X
Labelled_Data_Y = y_train[0:500, :]


# we need to compare binary images - instead of multi-label
# get image points with labels 1 and 2 only

# -- saving dataset into .mat
scipy.io.savemat('dataset.mat', dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test))


# we get the weights for all the data points - both labelled and unlabelled
#load uncertainty results

# weighted matrix W calculated on both labelled and unlabelled data - W is nxn

processed = scipy.io.loadmat('/Users/Riashat/Documents/Cambridge_THESIS/Experiments/keras/active_learning/Acquisition_Functions/SSL_Minimum_Bayes_Risk/Processed_Results.mat')
# All_Mean = uncertainty['All_Mean']
# All_Std = uncertainty['All_Std']
# Label_Prob = uncertainty['Label_Prob']
# Std = uncertainty['Std']
# Label_Class = uncertainty['Label_Class']
# Y_pred = uncertainty['Y_pred']

W = processed['W'] 
D = processed['D']
Delta = processed['Delta']

Delta_ll = processed['Delta_ll']
Delta_lu = processed['Delta_lu']
Delta_ul = processed['Delta_ul']
Delta_uu = processed['Delta_uu']

inv_Delta_uu = linalg.inv(Delta_uu)

f_L = Labelled_Data_Y

Delta_mult = np.dot(inv_Delta_uu, Delta_ul)
f_U = - np.dot(Delta_mult, f_L)				#f_U shape is 1500x1? - same as the number of unlabelled examples

f_I = np.concatenate((f_L, f_U), axis=0)





