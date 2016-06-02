# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)

Bayes_Segnet_Train_Loss = np.load('Segnet_Train.npy')
Bayes_Segnet_Valid_Loss = np.load('Segnet_Valid.npy')

plt.figure(figsize=(20, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Bayes_Segnet_Train_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Bayes_Segnet_Train_Loss[:,10],color="red", linewidth=1.0, marker='x',label="10th - Training")
plt.plot(epochs, Bayes_Segnet_Train_Loss[:,20],color="red", linewidth=1.0, marker='+',label="20th - Training")

plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th - Validation")
plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,20],color="black", linewidth=1.0, marker='+',label="20th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Dropout BAYES SEGNET - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.figure(figsize=(20, 10), dpi=80)
plt.figure(2)
plt.plot(epochs, Bayes_Segnet_Train_Loss, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Bayes_Segnet_Valid_Loss, color="black", linewidth=1.0, marker='+' )


plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Dropout BAYES SEGNET  - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.show()