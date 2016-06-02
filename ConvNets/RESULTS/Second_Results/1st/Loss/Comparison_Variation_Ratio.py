# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 21, 1)


Variation_Ratio_Train = np.load('Variation_Ratio_Train.npy')
Variation_Ratio_Valid = np.load('Variation_Ratio_Valid.npy')

plt.figure(figsize=(20, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Variation_Ratio_Train[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Variation_Ratio_Train[:,10],color="red", linewidth=1.0, marker='x',label="10th - Training")
plt.plot(epochs, Variation_Ratio_Train[:,20],color="red", linewidth=1.0, marker='+',label="20th - Training")

plt.plot(epochs, Variation_Ratio_Valid[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Variation_Ratio_Valid[:,10],color="black", linewidth=1.0, marker='x',label="10th - Validation")
plt.plot(epochs, Variation_Ratio_Valid[:,20],color="black", linewidth=1.0, marker='+',label="20th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Dropout VARIATION RATIO - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.figure(figsize=(20, 10), dpi=80)
plt.figure(2)
plt.plot(epochs, Variation_Ratio_Train, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Variation_Ratio_Valid, color="black", linewidth=1.0, marker='+' )


plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Dropout VARIATION RATIO - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.show()