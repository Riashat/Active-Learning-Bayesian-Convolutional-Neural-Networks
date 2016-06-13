# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, rcParams
import matplotlib.dates as dates

epochs = np.arange(1, 51, 1)


Variation_Ratio_Train = np.load('Variation_Ratio_Train.npy')
Variation_Ratio_Valid = np.load('Variation_Ratio_Valid.npy')




plt.figure(figsize=(15, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Variation_Ratio_Train[:,1], color="red", linewidth=2.0, linestyle='--',label="1st acquisition - Training")
plt.plot(epochs, Variation_Ratio_Train[:,10], color="black", linewidth=2.0, linestyle='--',label="30th acquisition - Training")
plt.plot(epochs, Variation_Ratio_Train[:,50],color="blue", linewidth=2.0, linestyle='--',label="50th acquisition - Training")
plt.plot(epochs, Variation_Ratio_Train[:,90],color="green", linewidth=2.0, linestyle='--',label="90th acquisition - Training")

plt.plot(epochs, Variation_Ratio_Valid[:,1], color="red", linewidth=2.0, linestyle='-',label="1st acquisition - Validation")
plt.plot(epochs, Variation_Ratio_Valid[:,10],color="black", linewidth=2.0,  linestyle='-', label="30th acquisition - Validation")
plt.plot(epochs, Variation_Ratio_Valid[:,50],color="blue", linewidth=2.0, linestyle='-',label="50th acquisition - Validation")
plt.plot(epochs, Variation_Ratio_Valid[:,90],color="green", linewidth=2.0, linestyle='-',label="90th acquisition - Validation")


plt.xlabel(r'\textbf{Number of Epochs}')
plt.ylabel(r'\textbf{Accuracy}')
plt.title(r'\textbf{Dropout Variation Ratio - Training and Validation Accuracy - Model Fitting}')
plt.grid()
plt.legend(loc = 4)
plt.show()





# plt.figure(figsize=(15, 10), dpi=80)

# plt.figure(1)

# plt.plot(epochs, Variation_Ratio_Train[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
# plt.plot(epochs, Variation_Ratio_Train[:,50],color="red", linewidth=1.0, marker='x',label="50th - Training")
# plt.plot(epochs, Variation_Ratio_Train[:,90],color="red", linewidth=1.0, marker='+',label="90th - Training")

# plt.plot(epochs, Variation_Ratio_Valid[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
# plt.plot(epochs, Variation_Ratio_Valid[:,90],color="black", linewidth=1.0, marker='x',label="50th - Validation")
# plt.plot(epochs, Variation_Ratio_Valid[:,50],color="black", linewidth=1.0, marker='+',label="90th - Validation")

# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy')
# plt.title('Dropout VARIATION RATIO - Training and Validation Accuracy')
# plt.grid()
# plt.legend(loc = 4)


# plt.figure(figsize=(15, 10), dpi=80)
# plt.figure(2)
# plt.plot(epochs, Variation_Ratio_Train, color="red", linewidth=1.0, marker='x')
# plt.plot(epochs, Variation_Ratio_Valid, color="black", linewidth=1.0, marker='+' )
# plt.plot(epochs, Variation_Ratio_Train[:, 50], color="red", linewidth=1.0, marker='x', label="Training Accuracy")
# plt.plot(epochs, Variation_Ratio_Valid[:, 50], color="black", linewidth=1.0, marker='+', label="Validation Accuracy" )

# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy')
# plt.title('Dropout VARIATION RATIO - Training and Validation Accuracy')
# plt.grid()
# plt.legend(loc = 4)


# plt.show()