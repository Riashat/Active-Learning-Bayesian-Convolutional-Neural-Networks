# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt



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


plt.figure(1)


plt.subplot(211)
plt.plot(epochs, Bald_Train_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Bald")
plt.plot(epochs, Bald_Train_Loss[:,10],color="red", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Bald")

plt.plot(epochs, Bayes_Segnet_Train_Loss[:,1], color="black", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Bayes Segnet")
plt.plot(epochs, Bayes_Segnet_Train_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Bayes Segnet")

plt.plot(epochs, Highest_Entropy_Train_Loss[:,1], color="blue", linewidth=1.0, marker='o',label="1st Acquisition - Max Entropy")
plt.plot(epochs, Highest_Entropy_Train_Loss[:,10],color="blue", linewidth=1.0, marker='x',label="10th Acquisition - Max Entropy")

plt.plot(epochs, BCNN_Max_Entropy_Train_Loss[:,1], color="green", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Max Entropy")
plt.plot(epochs, BCNN_Max_Entropy_Train_Loss[:,10],color="green", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Max Entropy")

plt.xlabel('Number of Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss at 1st and 10th Acquisition')
plt.grid()
plt.legend(loc = 1)





plt.subplot(212)
plt.plot(epochs, Bald_Valid_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Bald")
plt.plot(epochs, Bald_Valid_Loss[:,10],color="red", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Bald")

plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,1], color="black", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Bayes Segnet")
plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Bayes Segnet")

plt.plot(epochs, Highest_Entropy_Valid_Loss[:,1], color="blue", linewidth=1.0, marker='o',label="1st Acquisition - Max Entropy")
plt.plot(epochs, Highest_Entropy_Valid_Loss[:,10],color="blue", linewidth=1.0, marker='x',label="10th Acquisition - Max Entropy")

plt.plot(epochs, BCNN_Max_Entropy_Valid_Loss[:,1], color="green", linewidth=1.0, marker='o',label="1st Acquisition - Dropout Max Entropy")
plt.plot(epochs, BCNN_Max_Entropy_Valid_Loss[:,10],color="green", linewidth=1.0, marker='x',label="10th Acquisition - Dropout Max Entropy")

plt.xlabel('Number of Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss at 1st and 10th Acquisitions')
plt.grid()
plt.legend(loc = 1)




plt.show()



