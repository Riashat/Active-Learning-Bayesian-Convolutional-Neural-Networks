# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51, 1)


Variation_Ratio_Train = np.load('Variation_Ratio_Train.npy')
Variation_Ratio_Valid = np.load('Variation_Ratio_Valid.npy')

plt.figure(figsize=(15, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Variation_Ratio_Train[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Variation_Ratio_Train[:,20],color="red", linewidth=1.0, marker='x',label="20th - Training")
plt.plot(epochs, Variation_Ratio_Train[:,50],color="red", linewidth=1.0, marker='+',label="50th - Training")

plt.plot(epochs, Variation_Ratio_Valid[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Variation_Ratio_Valid[: 20],color="black", linewidth=1.0, marker='x',label="20th - Validation")
plt.plot(epochs, Variation_Ratio_Valid[:,50],color="black", linewidth=1.0, marker='+',label="50th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Dropout VARIATION RATIO - Training and Validation Accuracy')
plt.grid()
plt.legend(loc = 4)


plt.figure(figsize=(15, 10), dpi=80)
plt.figure(2)
plt.plot(epochs, Variation_Ratio_Train, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Variation_Ratio_Valid, color="black", linewidth=1.0, marker='+' )
plt.plot(epochs, Variation_Ratio_Train[:, 50], color="red", linewidth=1.0, marker='x', label="Training Accuracy")
plt.plot(epochs, Variation_Ratio_Valid[:, 50], color="black", linewidth=1.0, marker='+', label="Validation Accuracy" )

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Dropout VARIATION RATIO - Training and Validation Accuracy')
plt.grid()
plt.legend(loc = 4)


plt.show()