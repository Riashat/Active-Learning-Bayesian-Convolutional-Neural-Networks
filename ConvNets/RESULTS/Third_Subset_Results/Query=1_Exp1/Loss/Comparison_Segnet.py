# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)

Bayes_Segnet_Train_Loss = np.load('Segnet_Train.npy')
Bayes_Segnet_Valid_Loss = np.load('Segnet_Valid.npy')

plt.figure(figsize=(15, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Bayes_Segnet_Train_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Bayes_Segnet_Train_Loss[:,20],color="red", linewidth=1.0, marker='x',label="50th - Training")
plt.plot(epochs, Bayes_Segnet_Train_Loss[:,50],color="red", linewidth=1.0, marker='+',label="90th - Training")

plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,20],color="black", linewidth=1.0, marker='x',label="50th - Validation")
plt.plot(epochs, Bayes_Segnet_Valid_Loss[:,50],color="black", linewidth=1.0, marker='+',label="90th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Dropout BAYES SEGNET - Training and Validation Accuracy')
plt.grid()
plt.legend(loc = 4)


plt.figure(figsize=(15, 10), dpi=80)
plt.figure(2)
plt.plot(epochs, Bayes_Segnet_Train_Loss, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Bayes_Segnet_Valid_Loss, color="black", linewidth=1.0, marker='+' )
plt.plot(epochs, Bayes_Segnet_Train_Loss[:, 50], color="red", linewidth=1.0, marker='x', label="Training Accuracy")
plt.plot(epochs, Bayes_Segnet_Valid_Loss[:, 50], color="black", linewidth=1.0, marker='+', label="Validation Accuracy" )


plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Dropout BAYES SEGNET  - Training and Validation Accuracy')
plt.grid()
plt.legend(loc = 4)


plt.show()