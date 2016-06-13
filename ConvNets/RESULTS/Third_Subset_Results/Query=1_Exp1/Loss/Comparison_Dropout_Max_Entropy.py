# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)


Dropout_Max_Entropy_Train = np.load('Dropout_Max_Entropy_Train.npy')
Dropout_Max_Entropy_Valid = np.load('Dropout_Max_Entropy_Valid.npy')

plt.figure(figsize=(20, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Dropout_Max_Entropy_Train[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Dropout_Max_Entropy_Train[:,20],color="red", linewidth=1.0, marker='x',label="50th - Training")
plt.plot(epochs, Dropout_Max_Entropy_Train[:,50],color="red", linewidth=1.0, marker='+',label="90th - Training")

plt.plot(epochs, Dropout_Max_Entropy_Valid[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Dropout_Max_Entropy_Valid[:,20],color="black", linewidth=1.0, marker='x',label="50th - Validation")
plt.plot(epochs, Dropout_Max_Entropy_Valid[:,50],color="black", linewidth=1.0, marker='+',label="90th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Dropout MAX ENTROPY - Training and Validation Accuracy')
plt.grid()
plt.legend(loc = 4)


plt.figure(figsize=(20, 10), dpi=80)
plt.figure(2)
plt.plot(epochs, Dropout_Max_Entropy_Train, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Dropout_Max_Entropy_Valid, color="black", linewidth=1.0, marker='+' )
plt.plot(epochs, Dropout_Max_Entropy_Train[:, 50], color="red", linewidth=1.0, marker='x', label="Training Accuracy")
plt.plot(epochs, Dropout_Max_Entropy_Valid[:, 50], color="black", linewidth=1.0, marker='+', label="Validation Accuracy" )

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Dropout MAX ENTROPY- Training and Validation Accuracy')
plt.grid()
plt.legend(loc = 4)


plt.show()