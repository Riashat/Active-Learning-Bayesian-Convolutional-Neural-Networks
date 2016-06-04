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
from scipy.stats import mode



# input image dimensions
img_rows, img_cols = 28, 28


# the data, shuffled and split between tran and test sets
(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()

X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)




X_Pool = X_train_All[20000:60000, :, :, :]
y_Pool = y_train_All[20000:60000]

Total_Pooled_Images = 100

Bald_Pool = np.load('Bald_Pool.npy')

print('Pooling Dropout Bald Images')

#saving pooled images
for im in range(Total_Pooled_Images):
	Image = X_Pool[Bald_Pool[im], :, :, :]
	img = Image.reshape((28,28))
	sp.misc.imsave('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/RESULTS/Third_Subset_Results/Query=10/Pooled_Images/Bald_Pool_Images/'+'Pooled'+'_Image_'+str(im)+'.jpg', img)


Dropout_Max_Entropy_Pool = np.load('Dropout_Max_Entropy_Pool.npy')

print('Pooling Dropout Max Entropy Images')
#saving pooled images
for im in range(Total_Pooled_Images):
	Image = X_Pool[Dropout_Max_Entropy_Pool[im], :, :, :]
	img = Image.reshape((28,28))
	sp.misc.imsave('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/RESULTS/Third_Subset_Results/Query=10/Pooled_Images/Dropout_Max_Entropy_Images/'+'Pooled'+'_Image_'+str(im)+'.jpg', img)



# Segnet_Pool = np.load('Segnet_Pool.npy')

# print('Pooling Bayes Segnet Images')
# #saving pooled images
# for im in range(Total_Pooled_Images):
# 	Image = X_Pool[Segnet_Pool[im], :, :, :]
# 	img = Image.reshape((28,28))
# 	sp.misc.imsave('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/RESULTS/Cluster_Experiment_Results/2nd/Pooled_Images/Segnet_Pool_Images/'+'Pooled'+'_Image_'+str(im)+'.jpg', img)



Variation_Ratio_Pool = np.load('Variation_Ratio_Pool.npy')

print('Pooling Variation Ratio Images')

#saving pooled images
for im in range(Total_Pooled_Images):
	Image = X_Pool[Variation_Ratio_Pool[im], :, :, :]
	img = Image.reshape((28,28))
	sp.misc.imsave('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/RESULTS/Third_Subset_Results/Query=10/Pooled_Images/Variation_Ratio_Images/'+'Pooled'+'_Image_'+str(im)+'.jpg', img)



Max_Entropy_Pool = np.load('Max_Entropy_Pool.npy')

print('Pooling Max Entropy  Images')
#saving pooled images
for im in range(Total_Pooled_Images):
	Image = X_Pool[Max_Entropy_Pool[im], :, :, :]
	img = Image.reshape((28,28))
	sp.misc.imsave('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/RESULTS/Third_Subset_Results/Query=10/Pooled_Images/Max_Entropy_Images/'+'Pooled'+'_Image_'+str(im)+'.jpg', img)


Random_Pool = np.load('Random_Pool.npy')

print('Pooling Random Acquisition Images')
#saving pooled images
for im in range(Total_Pooled_Images):
	Image = X_Pool[Random_Pool[im], :, :, :]
	img = Image.reshape((28,28))
	sp.misc.imsave('/Users/Riashat/Documents/Cambridge_THESIS/Code/Experiments/keras/RESULTS/Third_Subset_Results/Query=10/Pooled_Images/Random_Images/'+'Pooled'+'_Image_'+str(im)+'.jpg', img)

