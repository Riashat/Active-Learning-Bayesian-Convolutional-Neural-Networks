# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)

Random_Train = np.load('Random_Train.npy')
Random_Valid = np.load('Random_Valid.npy')

plt.figure(figsize=(15, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Random_Train[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Random_Train[:,50],color="red", linewidth=1.0, marker='x',label="50th - Training")
plt.plot(epochs, Random_Train[:,90],color="red", linewidth=1.0, marker='+',label="90th - Training")

plt.plot(epochs, Random_Valid[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Random_Valid[:,50],color="black", linewidth=1.0, marker='x',label="50th - Validation")
plt.plot(epochs, Random_Valid[:,90],color="black", linewidth=1.0, marker='+',label="90th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Random Acquisitions  - Training and Validation Accuracy')
plt.grid()
plt.legend(loc = 4)


plt.figure(figsize=(15, 10), dpi=80)
plt.figure(2)
plt.plot(epochs, Random_Train, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Random_Valid, color="black", linewidth=1.0, marker='+' )
plt.plot(epochs, Random_Train[:, 50], color="red", linewidth=1.0, marker='x', label="Training Accuracy")
plt.plot(epochs, Random_Valid[:, 50], color="black", linewidth=1.0, marker='+', label="Validation Accuracy" )

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Random Acquisitions - Training and Validation Accuracy')
plt.grid()
plt.legend(loc = 4)


plt.show()