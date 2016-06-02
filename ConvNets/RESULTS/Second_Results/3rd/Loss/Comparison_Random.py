# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)

Random_Train = np.load('Random_Train.npy')
Random_Valid = np.load('Random_Valid.npy')

plt.figure(figsize=(20, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Random_Train[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Random_Train[:,10],color="red", linewidth=1.0, marker='x',label="10th - Training")
plt.plot(epochs, Random_Train[:,15],color="red", linewidth=1.0, marker='+',label="15th - Training")

plt.plot(epochs, Random_Valid[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Random_Valid[:,10],color="black", linewidth=1.0, marker='x',label="10th - Validation")
plt.plot(epochs, Random_Valid[:,15],color="black", linewidth=1.0, marker='+',label="15th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Random Acquisitions  - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.figure(figsize=(20, 10), dpi=80)
plt.figure(2)
plt.plot(epochs, Random_Train, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Random_Valid, color="black", linewidth=1.0, marker='+' )


plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Random Acquisitions - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.show()