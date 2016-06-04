# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)


Bald_Train_Loss = np.load('Subset_2nd_Train_Loss_Experiment_0.npy')
Bald_Valid_Loss = np.load('Subset_2nd_Valid_Loss_Experiment_0.npy')



Bald_Train_Loss = Bald_Train_Loss.reshape(Bald_Train_Loss.shape[1])
Bald_Valid_Loss = Bald_Valid_Loss.reshape(Bald_Valid_Loss.shape[1])


plt.figure(figsize=(20, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Bald_Train_Loss[0:50], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Bald_Train_Loss[200:250],color="red", linewidth=1.0, marker='+',label="5th - Training")
plt.plot(epochs, Bald_Train_Loss[500:550],color="red", linewidth=1.0, marker='x',label="10th - Training")


plt.plot(epochs, Bald_Valid_Loss[0:50], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Bald_Valid_Loss[200:250],color="black", linewidth=1.0, marker='x',label="5th - Validation")
plt.plot(epochs, Bald_Valid_Loss[500:550],color="black", linewidth=1.0, marker='+',label="10thth - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Dropout BALD - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


# plt.figure(figsize=(20, 10), dpi=80)
# plt.figure(2)
# plt.plot(epochs, Bald_Train_Loss, color="red", linewidth=1.0, marker='x')
# plt.plot(epochs, Bald_Valid_Loss, color="black", linewidth=1.0, marker='+' )


# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.title('Dropout BALD - Training and Validation Loss')
# plt.grid()
# plt.legend(loc = 1)


plt.show()