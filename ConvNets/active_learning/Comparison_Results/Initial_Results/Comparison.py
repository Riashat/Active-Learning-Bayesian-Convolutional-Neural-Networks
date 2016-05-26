# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt



Bald = np.load('Bald_All_Accuracy.npy')
Highest_Entropy = np.load('Highest_Entropy_All_Accuracy.npy')
Bayes_Segnet = np.load('Bayes_Segnet_All_Accuracy.npy')
BCNN_Entropy = np.load('BCNN_Only_Entropy_All_Accuracy.npy')


Bald_2 = np.load('2nd_Bald_all_accuracy.npy')
BCNN_Entropy_2 = np.load('2nd_BCNN_all_accuracy.npy')


Queries = np.arange(0, 1000, 100)

plt.figure(figsize=(15, 10), dpi=80)

Bald = np.array([Bald]).T
Bald_2 = np.array([Bald_2]).T
Bald_Avg = np.concatenate((Bald, Bald_2), axis=1)
Bald_Avg = np.mean(Bald_Avg, axis=1)

BCNN_Entropy = np.array([BCNN_Entropy]).T
BCNN_Entropy_2 = np.array([BCNN_Entropy_2]).T
BCNN_Entropy_Avg = np.concatenate((BCNN_Entropy, BCNN_Entropy_2), axis=1)
BCNN_Entropy_Avg = np.mean(BCNN_Entropy_Avg, axis=1)


plt.figure(1)
plt.plot(Queries, Bald_Avg[1:], color="red", linewidth=1.0, marker='x', label="Dropout BALD Average (2)" )
plt.plot(Queries, Bayes_Segnet[1:], color="blue", linewidth=1.0, marker='x', label="Dropout Bayes Segnet")
plt.plot(Queries, BCNN_Entropy_Avg[1:], color="black", linewidth=1.0, marker='x', label="Dropout - Max Entropy Average(2)")
plt.plot(Queries, Highest_Entropy[1:], color="green",linewidth=1.0, marker='x', label="Max Entropy" )

plt.xlabel('Number of Queries')
plt.ylabel('Accuracy on Test Set')
plt.title('Comparing Acquisition Functions')
plt.grid()
plt.legend(loc = 3)
plt.show()






epochs = np.arange(1, 11, 1)

Bald_Train_Loss = np.load('Bald_Pool_Train_Loss.npy')
Bald_Valid_Loss = np.load('Bald_Pool_Valid_Loss.npy')
Bald_Train_Loss_2 = np.load('2nd_BALD_Pool_Train_Loss.npy')
Bald_Valid_Loss_2 = np.load('2nd_BALD_Pool_Valid_Loss.npy')

Bayes_Segnet_Train_Loss = np.load('Bayes_Segnet_Pool_Train_Loss.npy')
Bayes_Segnet_Valid_Loss = np.load('Bayes_Segnet_Pool_Valid_Loss.npy')

Highest_Entropy_Train_Loss = np.load('Highest_Entropy_Pool_Train_Loss.npy')
Highest_Entropy_Valid_Loss = np.load('Highest_Entropy_Pool_Valid_Loss.npy')

BCNN_Max_Entropy_Train_Loss = np.load('BCNN_Only_Entropy_Pool_Train_Loss.npy')
BCNN_Max_Entropy_Valid_Loss = np.load('BCNN_Only_Entropy_Pool_Valid_Loss.npy')
BCNN_Max_Entropy_Train_Loss_2 = np.load('2nd_BCNN_Pool_Train_Loss.npy')
BCNN_Max_Entropy_Valid_Loss_2 = np.load('2nd_BCNN_Pool_Valid_Loss.npy')



plt.figure(figsize=(15, 10), dpi=80)
plt.figure(2)
plt.subplot(211)

plt.plot(epochs, Bald_Train_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Bald")
plt.plot(epochs, Bald_Train_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Bald")

plt.plot(epochs, Bayes_Segnet_Train_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Bayes Segnet")
plt.plot(epochs, Bayes_Segnet_Train_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Bayes Segnet")

plt.plot(epochs, Highest_Entropy_Train_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Max Entropy")
plt.plot(epochs, Highest_Entropy_Train_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Max Entropy")

plt.plot(epochs, BCNN_Max_Entropy_Train_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Max Entropy")
plt.plot(epochs, BCNN_Max_Entropy_Train_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Max Entropy")

plt.xlabel('Number of Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss at 1st and 10th Acquisition')
plt.grid()
plt.legend(loc = 1)


plt.subplot(212)
plt.plot(epochs, Bald_Valid_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Bald")
plt.plot(epochs, Bald_Valid_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Bald")

plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Bayes Segnet")
plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Bayes Segnet")

plt.plot(epochs, Highest_Entropy_Valid_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Max Entropy")
plt.plot(epochs, Highest_Entropy_Valid_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Max Entropy")

plt.plot(epochs, BCNN_Max_Entropy_Valid_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Max Entropy")
plt.plot(epochs, BCNN_Max_Entropy_Valid_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Max Entropy")

plt.xlabel('Number of Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss at 1st and 10th Acquisitions')
plt.grid()
plt.legend(loc = 1)


plt.show()



