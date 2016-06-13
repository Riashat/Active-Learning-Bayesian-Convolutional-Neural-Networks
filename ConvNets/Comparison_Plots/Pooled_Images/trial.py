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
# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between tran and test sets
(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()
X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_Pool = X_train_All[20000:60000, :, :, :]
y_Pool = y_train_All[20000:60000]


Total_Pooled_Images = 400



Bald_Pool = np.load('Bald_Pool.npy')

for im in range(10):
	Fig = X_Pool[Bald_Pool[1+im], :, :, :]
	img = Fig.reshape((28,28))
	sp.misc.imsave('Bald'+'_Image_'+str(im)+'.jpg', img)


fname = 'Bald_Image_0.jpg'
image = Image.open(fname).convert("L")
b1 = np.asarray(image)

fname = 'Bald_Image_1.jpg'
image = Image.open(fname).convert("L")
b2 = np.asarray(image)

fname = 'Bald_Image_2.jpg'
image = Image.open(fname).convert("L")
b3 = np.asarray(image)

fname = 'Bald_Image_3.jpg'
image = Image.open(fname).convert("L")
b4 = np.asarray(image)

fname = 'Bald_Image_4.jpg'
image = Image.open(fname).convert("L")
b5 = np.asarray(image)

fname = 'Bald_Image_5.jpg'
image = Image.open(fname).convert("L")
b6 = np.asarray(image)

fname = 'Bald_Image_6.jpg'
image = Image.open(fname).convert("L")
b7 = np.asarray(image)

fname = 'Bald_Image_7.jpg'
image = Image.open(fname).convert("L")
b8 = np.asarray(image)

fname = 'Bald_Image_8.jpg'
image = Image.open(fname).convert("L")
b9 = np.asarray(image)

fname = 'Bald_Image_9.jpg'
image = Image.open(fname).convert("L")
b10 = np.asarray(image)






Dropout_Max__Pool = np.load('Dropout_Max_Entropy_Pool.npy')

for im in range(10):
	Fig = X_Pool[Dropout_Max__Pool[1+im], :, :, :]
	img = Fig.reshape((28,28))
	sp.misc.imsave('Dropout_Max__Pool'+'_Image_'+str(im)+'.jpg', img)

fname = 'Dropout_Max__Pool_Image_0.jpg'
image = Image.open(fname).convert("L")
bm1 = np.asarray(image)

fname = 'Dropout_Max__Pool_Image_1.jpg'
image = Image.open(fname).convert("L")
bm2 = np.asarray(image)

fname = 'Dropout_Max__Pool_Image_2.jpg'
image = Image.open(fname).convert("L")
bm3 = np.asarray(image)

fname = 'Dropout_Max__Pool_Image_3.jpg'
image = Image.open(fname).convert("L")
bm4 = np.asarray(image)

fname = 'Dropout_Max__Pool_Image_4.jpg'
image = Image.open(fname).convert("L")
bm5 = np.asarray(image)

fname = 'Dropout_Max__Pool_Image_5.jpg'
image = Image.open(fname).convert("L")
bm6 = np.asarray(image)

fname = 'Dropout_Max__Pool_Image_6.jpg'
image = Image.open(fname).convert("L")
bm7 = np.asarray(image)

fname = 'Dropout_Max__Pool_Image_7.jpg'
image = Image.open(fname).convert("L")
bm8 = np.asarray(image)

fname = 'Dropout_Max__Pool_Image_8.jpg'
image = Image.open(fname).convert("L")
bm9 = np.asarray(image)

fname = 'Dropout_Max__Pool_Image_9.jpg'
image = Image.open(fname).convert("L")
bm10 = np.asarray(image)




# Segnet_Pool = np.load('Segnet_Pool.npy')

# for im in range(10):
# 	Fig = X_Pool[Segnet_Pool[1+im], :, :, :]
# 	img = Fig.reshape((28,28))
# 	sp.misc.imsave('Segnet_Pool'+'_Image_'+str(im)+'.jpg', img)

# fname = 'Segnet_Pool_Image_0.jpg'
# image = Image.open(fname).convert("L")
# s1 = np.asarray(image)

# fname = 'Segnet_Pool_Image_1.jpg'
# image = Image.open(fname).convert("L")
# s2 = np.asarray(image)

# fname = 'Segnet_Pool_Image_2.jpg'
# image = Image.open(fname).convert("L")
# s3 = np.asarray(image)

# fname = 'Segnet_Pool_Image_3.jpg'
# image = Image.open(fname).convert("L")
# s4 = np.asarray(image)

# fname = 'Segnet_Pool_Image_4.jpg'
# image = Image.open(fname).convert("L")
# s5 = np.asarray(image)

# fname = 'Segnet_Pool_Image_5.jpg'
# image = Image.open(fname).convert("L")
# s6 = np.asarray(image)

# fname = 'Segnet_Pool_Image_6.jpg'
# image = Image.open(fname).convert("L")
# s7 = np.asarray(image)

# fname = 'Segnet_Pool_Image_7.jpg'
# image = Image.open(fname).convert("L")
# s8 = np.asarray(image)

# fname = 'Segnet_Pool_Image_8.jpg'
# image = Image.open(fname).convert("L")
# s9 = np.asarray(image)

# fname = 'Segnet_Pool_Image_9.jpg'
# image = Image.open(fname).convert("L")
# s10 = np.asarray(image)




Var_Pool = np.load('Variation_Ratio_Pool.npy')

for im in range(10):
	Fig = X_Pool[Var_Pool[1+im], :, :, :]
	img = Fig.reshape((28,28))
	sp.misc.imsave('Var_Pool'+'_Image_'+str(im)+'.jpg', img)

fname = 'Var_Pool_Image_0.jpg'
image = Image.open(fname).convert("L")
v1 = np.asarray(image)

fname = 'Var_Pool_Image_1.jpg'
image = Image.open(fname).convert("L")
v2 = np.asarray(image)

fname = 'Var_Pool_Image_2.jpg'
image = Image.open(fname).convert("L")
v3 = np.asarray(image)

fname = 'Var_Pool_Image_3.jpg'
image = Image.open(fname).convert("L")
v4 = np.asarray(image)

fname = 'Var_Pool_Image_4.jpg'
image = Image.open(fname).convert("L")
v5 = np.asarray(image)

fname = 'Var_Pool_Image_5.jpg'
image = Image.open(fname).convert("L")
v6 = np.asarray(image)

fname = 'Var_Pool_Image_6.jpg'
image = Image.open(fname).convert("L")
v7 = np.asarray(image)

fname = 'Var_Pool_Image_7.jpg'
image = Image.open(fname).convert("L")
v8 = np.asarray(image)

fname = 'Var_Pool_Image_8.jpg'
image = Image.open(fname).convert("L")
v9 = np.asarray(image)

fname = 'Var_Pool_Image_9.jpg'
image = Image.open(fname).convert("L")
v10 = np.asarray(image)








Max_Pool = np.load('Max_Entropy_Pool.npy')

for im in range(10):
	Fig = X_Pool[Max_Pool[1+im], :, :, :]
	img = Fig.reshape((28,28))
	sp.misc.imsave('Max_Pool'+'_Image_'+str(im)+'.jpg', img)

fname = 'Max_Pool_Image_0.jpg'
image = Image.open(fname).convert("L")
m1 = np.asarray(image)

fname = 'Max_Pool_Image_1.jpg'
image = Image.open(fname).convert("L")
m2 = np.asarray(image)

fname = 'Max_Pool_Image_2.jpg'
image = Image.open(fname).convert("L")
m3 = np.asarray(image)

fname = 'Max_Pool_Image_3.jpg'
image = Image.open(fname).convert("L")
m4 = np.asarray(image)

fname = 'Max_Pool_Image_4.jpg'
image = Image.open(fname).convert("L")
m5 = np.asarray(image)

fname = 'Max_Pool_Image_5.jpg'
image = Image.open(fname).convert("L")
m6 = np.asarray(image)

fname = 'Max_Pool_Image_6.jpg'
image = Image.open(fname).convert("L")
m7 = np.asarray(image)

fname = 'Max_Pool_Image_7.jpg'
image = Image.open(fname).convert("L")
m8 = np.asarray(image)

fname = 'Max_Pool_Image_8.jpg'
image = Image.open(fname).convert("L")
m9 = np.asarray(image)

fname = 'Max_Pool_Image_9.jpg'
image = Image.open(fname).convert("L")
m10 = np.asarray(image)









Rand_Pool = np.load('Random_Pool.npy')

for im in range(10):
	Fig = X_Pool[Rand_Pool[1+im], :, :, :]
	img = Fig.reshape((28,28))
	sp.misc.imsave('Rand_Pool'+'_Image_'+str(im)+'.jpg', img)

fname = 'Rand_Pool_Image_0.jpg'
image = Image.open(fname).convert("L")
r1 = np.asarray(image)

fname = 'Rand_Pool_Image_1.jpg'
image = Image.open(fname).convert("L")
r2 = np.asarray(image)

fname = 'Rand_Pool_Image_2.jpg'
image = Image.open(fname).convert("L")
r3 = np.asarray(image)

fname = 'Rand_Pool_Image_3.jpg'
image = Image.open(fname).convert("L")
r4 = np.asarray(image)

fname = 'Rand_Pool_Image_4.jpg'
image = Image.open(fname).convert("L")
r5 = np.asarray(image)

fname = 'Rand_Pool_Image_5.jpg'
image = Image.open(fname).convert("L")
r6 = np.asarray(image)

fname = 'Rand_Pool_Image_6.jpg'
image = Image.open(fname).convert("L")
r7 = np.asarray(image)

fname = 'Rand_Pool_Image_7.jpg'
image = Image.open(fname).convert("L")
r8 = np.asarray(image)

fname = 'Rand_Pool_Image_8.jpg'
image = Image.open(fname).convert("L")
r9 = np.asarray(image)

fname = 'Rand_Pool_Image_9.jpg'
image = Image.open(fname).convert("L")
r10 = np.asarray(image)



plt.figure(figsize=(10, 10), dpi=80)
f, axarr = plt.subplots(5,10)


axarr[0, 0].imshow(b1, cmap='Greys_r')
axarr[0, 1].imshow(b2, cmap='Greys_r')
axarr[0, 2].imshow(b3, cmap='Greys_r')
axarr[0, 3].imshow(b4, cmap='Greys_r')
axarr[0, 4].imshow(b5, cmap='Greys_r')
axarr[0, 5].imshow(b6, cmap='Greys_r')
axarr[0, 6].imshow(b7, cmap='Greys_r')
axarr[0, 7].imshow(b8, cmap='Greys_r')
axarr[0, 8].imshow(b9, cmap='Greys_r')
axarr[0, 9].imshow(b10, cmap='Greys_r')
axarr[0, 0].set_title('Dropout BALD')
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[0, :]], visible=False)

axarr[1, 0].imshow(bm1, cmap='Greys_r')
axarr[1, 1].imshow(bm2, cmap='Greys_r')
axarr[1, 2].imshow(bm3, cmap='Greys_r')
axarr[1, 3].imshow(bm4, cmap='Greys_r')
axarr[1, 4].imshow(bm5, cmap='Greys_r')
axarr[1, 5].imshow(bm6, cmap='Greys_r')
axarr[1, 6].imshow(bm7, cmap='Greys_r')
axarr[1, 7].imshow(bm8, cmap='Greys_r')
axarr[1, 8].imshow(bm9, cmap='Greys_r')
axarr[1, 9].imshow(bm10, cmap='Greys_r')
axarr[1, 0].set_title('Dropout Max Entropy')
plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[1, :]], visible=False)


axarr[2, 0].imshow(v1, cmap='Greys_r')
axarr[2, 1].imshow(v2, cmap='Greys_r')
axarr[2, 2].imshow(v3, cmap='Greys_r')
axarr[2, 3].imshow(v4, cmap='Greys_r')
axarr[2, 4].imshow(v5, cmap='Greys_r')
axarr[2, 5].imshow(v6, cmap='Greys_r')
axarr[2, 6].imshow(v7, cmap='Greys_r')
axarr[2, 7].imshow(v8, cmap='Greys_r')
axarr[2, 8].imshow(v9, cmap='Greys_r')
axarr[2, 9].imshow(v10, cmap='Greys_r')
axarr[2, 0].set_title('Dropout Variation Ratio')
plt.setp([a.get_xticklabels() for a in axarr[2, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[2, :]], visible=False)



axarr[3, 0].imshow(m1, cmap='Greys_r')
axarr[3, 1].imshow(m2, cmap='Greys_r')
axarr[3, 2].imshow(m3, cmap='Greys_r')
axarr[3, 3].imshow(m4, cmap='Greys_r')
axarr[3, 4].imshow(m5, cmap='Greys_r')
axarr[3, 5].imshow(m6, cmap='Greys_r')
axarr[3, 6].imshow(m7, cmap='Greys_r')
axarr[3, 7].imshow(m8, cmap='Greys_r')
axarr[3, 8].imshow(m9, cmap='Greys_r')
axarr[3, 9].imshow(m10, cmap='Greys_r')
axarr[3, 0].set_title('Max Entropy')
plt.setp([a.get_xticklabels() for a in axarr[3, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[3, :]], visible=False)


axarr[4, 0].imshow(r1, cmap='Greys_r')
axarr[4, 1].imshow(r2, cmap='Greys_r')
axarr[4, 2].imshow(r3, cmap='Greys_r')
axarr[4, 3].imshow(r4, cmap='Greys_r')
axarr[4, 4].imshow(r5, cmap='Greys_r')
axarr[4, 5].imshow(r6, cmap='Greys_r')
axarr[4, 6].imshow(r7, cmap='Greys_r')
axarr[4, 7].imshow(r8, cmap='Greys_r')
axarr[4, 8].imshow(r9, cmap='Greys_r')
axarr[4, 9].imshow(r10, cmap='Greys_r')
axarr[4, 0].set_title('Random')
plt.setp([a.get_xticklabels() for a in axarr[4, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[4, :]], visible=False)


plt.show()




'''


plt.figure(figsize=(10, 10), dpi=80)
f, axarr = plt.subplots(6,10)


axarr[0, 0].imshow(b1, cmap='Greys_r')
axarr[0, 1].imshow(b2, cmap='Greys_r')
axarr[0, 2].imshow(b3, cmap='Greys_r')
axarr[0, 3].imshow(b4, cmap='Greys_r')
axarr[0, 4].imshow(b5, cmap='Greys_r')
axarr[0, 5].imshow(b6, cmap='Greys_r')
axarr[0, 6].imshow(b7, cmap='Greys_r')
axarr[0, 7].imshow(b8, cmap='Greys_r')
axarr[0, 8].imshow(b9, cmap='Greys_r')
axarr[0, 9].imshow(b10, cmap='Greys_r')

axarr[0, 0].set_title('Sharing x per column, y per row')
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[0, :]], visible=False)

axarr[1, 0].imshow(bm1, cmap='Greys_r')
axarr[1, 1].imshow(bm2, cmap='Greys_r')
axarr[1, 2].imshow(bm3, cmap='Greys_r')
axarr[1, 3].imshow(bm4, cmap='Greys_r')
axarr[1, 4].imshow(bm5, cmap='Greys_r')
axarr[1, 5].imshow(bm6, cmap='Greys_r')
axarr[1, 6].imshow(bm7, cmap='Greys_r')
axarr[1, 7].imshow(bm8, cmap='Greys_r')
axarr[1, 8].imshow(bm9, cmap='Greys_r')
axarr[1, 9].imshow(bm10, cmap='Greys_r')
axarr[1, 0].set_title('Sharing x per column, y per row')
plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[1, :]], visible=False)

axarr[2, 0].imshow(s1, cmap='Greys_r')
axarr[2, 1].imshow(s2, cmap='Greys_r')
axarr[2, 2].imshow(s3, cmap='Greys_r')
axarr[2, 3].imshow(s4, cmap='Greys_r')
axarr[2, 4].imshow(s5, cmap='Greys_r')
axarr[2, 5].imshow(s6, cmap='Greys_r')
axarr[2, 6].imshow(s7, cmap='Greys_r')
axarr[2, 7].imshow(s8, cmap='Greys_r')
axarr[2, 8].imshow(s9, cmap='Greys_r')
axarr[2, 9].imshow(s10, cmap='Greys_r')
axarr[2, 0].set_title('Sharing x per column, y per row')
plt.setp([a.get_xticklabels() for a in axarr[2, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[2, :]], visible=False)


axarr[3, 0].imshow(v1, cmap='Greys_r')
axarr[3, 1].imshow(v2, cmap='Greys_r')
axarr[3, 2].imshow(v3, cmap='Greys_r')
axarr[3, 3].imshow(v4, cmap='Greys_r')
axarr[3, 4].imshow(v5, cmap='Greys_r')
axarr[3, 5].imshow(v6, cmap='Greys_r')
axarr[3, 6].imshow(v7, cmap='Greys_r')
axarr[3, 7].imshow(v8, cmap='Greys_r')
axarr[3, 8].imshow(v9, cmap='Greys_r')
axarr[3, 9].imshow(v10, cmap='Greys_r')
axarr[3, 0].set_title('Sharing x per column, y per row')
plt.setp([a.get_xticklabels() for a in axarr[3, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[3, :]], visible=False)



axarr[4, 0].imshow(m1, cmap='Greys_r')
axarr[4, 1].imshow(m2, cmap='Greys_r')
axarr[4, 2].imshow(m3, cmap='Greys_r')
axarr[4, 3].imshow(m4, cmap='Greys_r')
axarr[4, 4].imshow(m5, cmap='Greys_r')
axarr[4, 5].imshow(m6, cmap='Greys_r')
axarr[4, 6].imshow(m7, cmap='Greys_r')
axarr[4, 7].imshow(m8, cmap='Greys_r')
axarr[4, 8].imshow(m9, cmap='Greys_r')
axarr[4, 9].imshow(m10, cmap='Greys_r')
axarr[4, 0].set_title('Sharing x per column, y per row')
plt.setp([a.get_xticklabels() for a in axarr[4, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[4, :]], visible=False)


axarr[5, 0].imshow(r1, cmap='Greys_r')
axarr[5, 1].imshow(r2, cmap='Greys_r')
axarr[5, 2].imshow(r3, cmap='Greys_r')
axarr[5, 3].imshow(r4, cmap='Greys_r')
axarr[5, 4].imshow(r5, cmap='Greys_r')
axarr[5, 5].imshow(r6, cmap='Greys_r')
axarr[5, 6].imshow(r7, cmap='Greys_r')
axarr[5, 7].imshow(r8, cmap='Greys_r')
axarr[5, 8].imshow(r9, cmap='Greys_r')
axarr[5, 9].imshow(r10, cmap='Greys_r')
axarr[5, 0].set_title('Sharing x per column, y per row')
plt.setp([a.get_xticklabels() for a in axarr[5, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[5, :]], visible=False)



plt.show()

'''

