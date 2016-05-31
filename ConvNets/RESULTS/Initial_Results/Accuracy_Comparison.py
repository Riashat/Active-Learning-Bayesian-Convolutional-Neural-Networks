# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt



Bald = np.load('Bald_All_Accuracy.npy')
Highest_Entropy = np.load('Highest_Entropy_All_Accuracy.npy')
Bayes_Segnet = np.load('Bayes_Segnet_All_Accuracy.npy')
BCNN_Entropy = np.load('BCNN_Only_Entropy_All_Accuracy.npy')

Random = np.load('Random_Accuracy.npy')

Variation = np.load('Variation_Ratio_Accuracy_Results.npy')


Bald_2 = np.load('2nd_Bald_all_accuracy.npy')
BCNN_Entropy_2 = np.load('2nd_BCNN_all_accuracy.npy')


Queries = np.arange(100, 1100, 100)




plt.figure(figsize=(12, 8), dpi=80)


Bald = np.array([Bald]).T
Bald_2 = np.array([Bald_2]).T
Bald_Avg = np.concatenate((Bald, Bald_2), axis=1)
Bald_Avg = np.mean(Bald_Avg, axis=1)



BCNN_Entropy = np.array([BCNN_Entropy]).T
BCNN_Entropy_2 = np.array([BCNN_Entropy_2]).T
BCNN_Entropy_Avg = np.concatenate((BCNN_Entropy, BCNN_Entropy_2), axis=1)
BCNN_Entropy_Avg = np.mean(BCNN_Entropy_Avg, axis=1)



plt.plot(Queries, Bald_Avg[1:], color="red", linewidth=1.0, marker='x', label="Dropout BALD Average (2)" )

plt.plot(Queries, Bayes_Segnet[1:], color="blue", linewidth=1.0, marker='x', label="Dropout Bayes Segnet")

plt.plot(Queries, BCNN_Entropy_Avg[1:], color="black", linewidth=1.0, marker='x', label="Dropout - Max Entropy Average(2)")

plt.plot(Queries, Highest_Entropy[1:], color="green",linewidth=1.0, marker='x', label="Max Entropy" )

plt.plot(Queries, Random[1:], color="magenta",linewidth=1.0, marker='x', label="Random Acquisition" )

plt.plot(Queries, Variation[1:], color="yellow",linewidth=1.0, marker='x', label="Dropout - Variation Ratio Acquisition" )


plt.xlabel('Number of Queries')
plt.ylabel('Accuracy on Test Set')
plt.title('Comparing Acquisition Functions')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)

# plt.ylim(0.8, 1.0)
plt.legend(loc = 3)
plt.show()
