# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)


Dropout_Max_Entropy_Train = np.load('Dropout_Max_Entropy_Train.npy')
Dropout_Max_Entropy_Valid = np.load('Dropout_Max_Entropy_Valid.npy')

Dropout_Max_Entropy_Train = Dropout_Max_Entropy_Train.reshape(Dropout_Max_Entropy_Train.shape[1])
Dropout_Max_Entropy_Valid = Dropout_Max_Entropy_Valid.reshape(Dropout_Max_Entropy_Valid.shape[1])

plt.figure(figsize=(20, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Dropout_Max_Entropy_Train[0:50], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Dropout_Max_Entropy_Train[200:250],color="red", linewidth=1.0, marker='x',label="5th - Training")
plt.plot(epochs, Dropout_Max_Entropy_Train[500:550],color="red", linewidth=1.0, marker='+',label="10th - Training")

plt.plot(epochs, Dropout_Max_Entropy_Valid[0:50], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Dropout_Max_Entropy_Valid[200:250],color="black", linewidth=1.0, marker='x',label="5th - Validation")
plt.plot(epochs, Dropout_Max_Entropy_Valid[500:550],color="black", linewidth=1.0, marker='+',label="10th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Dropout MAX ENTROPY - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.figure(figsize=(20, 10), dpi=80)
plt.figure(2)
plt.plot(epochs, Dropout_Max_Entropy_Train, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Dropout_Max_Entropy_Valid, color="black", linewidth=1.0, marker='+' )


plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Dropout MAX ENTROPY- Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.show()