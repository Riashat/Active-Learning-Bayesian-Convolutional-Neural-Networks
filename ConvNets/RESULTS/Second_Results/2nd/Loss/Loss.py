# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)

Bald_Train_Loss = np.load('Bald_Train.npy')
Bayes_Segnet_Train_Loss = np.load('Segnet_Train.npy')
Dropout_Max_Entropy_Train = np.load('Dropout_Max_Entropy_Train.npy')
Variation_Ratio_Train = np.load('Variation_Ratio_Train.npy')
Highest_Entropy_Train_Loss = np.load('Max_Entropy_Train.npy')
Random_Train = np.load('Random_Train.npy')


plt.figure(figsize=(20, 10), dpi=80)

plt.figure(1)
# plt.subplot(211)

plt.plot(epochs, Bald_Train_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st - Dropout Bald")
plt.plot(epochs, Bald_Train_Loss[:,10],color="red", linewidth=1.0, marker='x',label="10th - Dropout Bald")
plt.plot(epochs, Bald_Train_Loss[:,20],color="red", linewidth=1.0, marker='+',label="20th - Dropout Bald")

plt.plot(epochs, Bayes_Segnet_Train_Loss[:,1], color="black", linewidth=1.0, marker='o',label="1st - Dropout Bayes Segnet")
plt.plot(epochs, Bayes_Segnet_Train_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th - Dropout Bayes Segnet")
plt.plot(epochs, Bayes_Segnet_Train_Loss[:,20],color="black", linewidth=1.0, marker='+',label="20th - Dropout Bayes Segnet")

plt.plot(epochs, Dropout_Max_Entropy_Train[:,1], color="green", linewidth=1.0, marker='o',label="1st - Dropout Max Entropy")
plt.plot(epochs, Dropout_Max_Entropy_Train[:,10],color="green", linewidth=1.0, marker='x',label="10th - Dropout Max Entropy")
plt.plot(epochs, Dropout_Max_Entropy_Train[:,20],color="green", linewidth=1.0, marker='+',label="20th - Dropout Max Entropy")

plt.plot(epochs, Variation_Ratio_Train[:,1], color="magenta", linewidth=1.0, marker='o',label="1st - Dropout Variation Ratio")
plt.plot(epochs, Variation_Ratio_Train[:,10],color="magenta", linewidth=1.0, marker='x',label="10th - Dropout Variation Ratio")
plt.plot(epochs, Variation_Ratio_Train[:,20],color="magenta", linewidth=1.0, marker='+',label="20th - Dropout Variation Ratio")

plt.plot(epochs, Highest_Entropy_Train_Loss[:,1], color="blue", linewidth=1.0, marker='o',label="1st - Max Entropy")
plt.plot(epochs, Highest_Entropy_Train_Loss[:,10],color="blue", linewidth=1.0, marker='x',label="10th - Max Entropy")
plt.plot(epochs, Highest_Entropy_Train_Loss[:,20],color="blue", linewidth=1.0, marker='+',label="20th - Max Entropy")

plt.plot(epochs, Random_Train[:,1], color="yellow", linewidth=1.0, marker='o',label="1st Acquisition - Random")
plt.plot(epochs, Random_Train[:,10],color="yellow", linewidth=1.0, marker='x',label="10th Acquisition - Random")
plt.plot(epochs, Random_Train[:,20],color="yellow", linewidth=1.0, marker='+',label="20th Acquisition - Random")

plt.xlabel('Number of Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss at 1st and 10th Acquisition')
plt.grid()
plt.legend(loc = 1)


Bald_Valid_Loss = np.load('Bald_Valid.npy')
Bayes_Segnet_Valid_Loss = np.load('Segnet_Valid.npy')
Dropout_Max_Entropy_Valid = np.load('Dropout_Max_Entropy_Valid.npy')
Variation_Ratio_Valid = np.load('Variation_Ratio_Valid.npy')
Highest_Entropy_Valid_Loss = np.load('Max_Entropy_Valid.npy')
Random_Valid = np.load('Random_Valid.npy')



plt.figure(figsize=(20, 10), dpi=80)
plt.figure(2)
# plt.subplot(212)
plt.plot(epochs, Bald_Valid_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st - Dropout Bald")
plt.plot(epochs, Bald_Valid_Loss[:,10],color="red", linewidth=1.0, marker='x',label="10th - Dropout Bald")
plt.plot(epochs, Bald_Valid_Loss[:,20],color="red", linewidth=1.0, marker='+',label="20th - Dropout Bald")

plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,1], color="black", linewidth=1.0, marker='o',label="1st - Dropout Bayes Segnet")
plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th - Dropout Bayes Segnet")
plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,20],color="black", linewidth=1.0, marker='+',label="20th - Dropout Bayes Segnet")

plt.plot(epochs, Dropout_Max_Entropy_Valid[:,1], color="green", linewidth=1.0, marker='o',label="1st - Dropout Max Entropy")
plt.plot(epochs, Dropout_Max_Entropy_Valid[:,10],color="green", linewidth=1.0, marker='x',label="10th - Dropout Max Entropy")
plt.plot(epochs, Dropout_Max_Entropy_Valid[:,20],color="green", linewidth=1.0, marker='+',label="20th - Dropout Max Entropy")

plt.plot(epochs, Variation_Ratio_Valid[:,1], color="magenta", linewidth=1.0, marker='o',label="1st - Dropout Variation Ratio")
plt.plot(epochs, Variation_Ratio_Valid[:,10],color="magenta", linewidth=1.0, marker='x',label="10th - Dropout Variation Ratio")
plt.plot(epochs, Variation_Ratio_Valid[:,20],color="magenta", linewidth=1.0, marker='+',label="20th - Dropout Variation Ratio")

plt.plot(epochs, Highest_Entropy_Valid_Loss[:,1], color="blue", linewidth=1.0, marker='o',label="1st - Max Entropy")
plt.plot(epochs, Highest_Entropy_Valid_Loss[:,10],color="blue", linewidth=1.0, marker='x',label="10th - Max Entropy")
plt.plot(epochs, Highest_Entropy_Valid_Loss[:,20],color="blue", linewidth=1.0, marker='+',label="20th - Max Entropy")

plt.plot(epochs, Random_Valid[:,1], color="yellow", linewidth=1.0, marker='o',label="1st - Random")
plt.plot(epochs, Random_Valid[:,10],color="yellow", linewidth=1.0, marker='x',label="10th - Random")
plt.plot(epochs, Random_Valid[:,20],color="yellow", linewidth=1.0, marker='+',label="20th - Random")
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss at 1st and 10th Acquisitions')
plt.grid()
plt.legend(loc = 1)


plt.show()
