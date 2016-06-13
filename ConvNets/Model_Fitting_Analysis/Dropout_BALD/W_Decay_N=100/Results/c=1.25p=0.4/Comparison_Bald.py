# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)


Bald_Train_Loss = np.load('Bald_Train.npy')
Bald_Valid_Loss = np.load('Bald_Valid.npy')


plt.figure(figsize=(15, 10), dpi=80)
plt.figure(1)
plt.plot(epochs, Bald_Train_Loss, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Bald_Valid_Loss, color="black", linewidth=1.0, marker='+')
plt.plot(epochs, Bald_Train_Loss[:, 16], color="red", linewidth=1.0, marker='x', label="Training Accuracy")
plt.plot(epochs, Bald_Valid_Loss[:, 16], color="black", linewidth=1.0, marker='+', label="Validation Accuracy" )


plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Dropout BALD - Training and Validation Accuracy')
plt.grid()
plt.legend(loc = 4)


plt.show()