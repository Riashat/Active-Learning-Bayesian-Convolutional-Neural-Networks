# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt


Bald = np.load('BALD.npy')
Highest_Entropy = np.load('Max_Entropy.npy')
Bayes_Segnet = np.load('Segnet.npy')
BCNN_Entropy = np.load('Dropout_Max_Entropy.npy')
Random = np.load('Random.npy')
Variation = np.load('Variation_Ratio.npy')

N100 = np.arange(50, 110, 10)
N1000 = np.arange(50, 1010, 10)


#0-5 - the first 6 points = total N=100

Bald_100 = Bald[0:6]
Bayes_Segnet_100 = Bayes_Segnet[0:6]
BCNN_Entropy_100 = BCNN_Entropy[0:6]
Variation_100 = Variation[0:6]


plt.figure(figsize=(15, 10), dpi=80)

plt.subplot(211)
plt.plot(N100, Bald_100, color="red", linewidth=1.0, marker='o', label="Dropout BALD" )
plt.plot(N100, Bayes_Segnet_100, color="black", linewidth=1.0, marker='o', label="Dropout Bayes Segnet" )
plt.plot(N100, BCNN_Entropy_100, color="blue", linewidth=1.0, marker='o', label="Dropout - Max Entropy" )
plt.plot(N100, Variation_100, color="green", linewidth=1.0, marker='o', label="Dropout - Variation Ratio")

plt.xlabel('Total Labelled Samples')
plt.ylabel('Accuracy on Test Set')
plt.title('100 Labelled Samples, Query=10')
plt.grid()
plt.legend(loc = 4)

plt.subplot(212)
plt.plot(N1000, Bald, color="red", linewidth=1.0, marker='o', label="Dropout BALD" )
plt.plot(N1000, Bayes_Segnet, color="black", linewidth=1.0, marker='o', label="Dropout Bayes Segnet" )
plt.plot(N1000, BCNN_Entropy, color="blue", linewidth=1.0, marker='o', label="Dropout - Max Entropy" )
plt.plot(N1000, Variation, color="green", linewidth=1.0, marker='o', label="Dropout - Variation Ratio")

plt.xlabel('Total Labelled Samples')
plt.ylabel('Accuracy on Test Set')
plt.title('1000 Labelled Samples, Query=10')
plt.grid()
plt.legend(loc = 4)
plt.show()
