# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)

Random_Train = np.load('Random_Train.npy')
Random_Valid = np.load('Random_Valid.npy')


Random_Train = Random_Train.reshape(Random_Train.shape[1])
Random_Valid = Random_Valid.reshape(Random_Valid.shape[1])




plt.figure(figsize=(20, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Random_Train[0:50], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Random_Train[200:250],color="red", linewidth=1.0, marker='x',label="10th - Training")
plt.plot(epochs, Random_Train[500:550],color="red", linewidth=1.0, marker='+',label="20th - Training")

plt.plot(epochs, Random_Valid[0:50], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Random_Valid[200:250],color="black", linewidth=1.0, marker='x',label="10th - Validation")
plt.plot(epochs, Random_Valid[500:550],color="black", linewidth=1.0, marker='+',label="20th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Random Acquisitions  - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


# plt.figure(figsize=(20, 10), dpi=80)
# plt.figure(2)
# plt.plot(epochs, Random_Train, color="red", linewidth=1.0, marker='x')
# plt.plot(epochs, Random_Valid, color="black", linewidth=1.0, marker='+' )


# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.title('Random Acquisitions - Training and Validation Loss')
# plt.grid()
# plt.legend(loc = 1)


plt.show()