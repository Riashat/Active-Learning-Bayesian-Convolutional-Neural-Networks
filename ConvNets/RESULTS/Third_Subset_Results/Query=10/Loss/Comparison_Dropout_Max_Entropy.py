# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, rcParams
import matplotlib.dates as dates

  # activate latex text rendering
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

epochs = np.arange(1, 51, 1)


Dropout_Max_Entropy_Train = np.load('Dropout_Max_Entropy_Train.npy')
Dropout_Max_Entropy_Valid = np.load('Dropout_Max_Entropy_Valid.npy')


plt.figure(figsize=(15, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Dropout_Max_Entropy_Train[:,1], color="red", linewidth=2.0, linestyle='--',label="1st acquisition - Training")
plt.plot(epochs, Dropout_Max_Entropy_Train[:,10], color="black", linewidth=2.0, linestyle='--',label="30th acquisition - Training")
plt.plot(epochs, Dropout_Max_Entropy_Train[:,50],color="blue", linewidth=2.0, linestyle='--',label="50th acquisition - Training")
plt.plot(epochs, Dropout_Max_Entropy_Train[:,90],color="green", linewidth=2.0, linestyle='--',label="90th acquisition - Training")

plt.plot(epochs, Dropout_Max_Entropy_Valid[:,1], color="red", linewidth=2.0, linestyle='-',label="1st acquisition - Validation")
plt.plot(epochs, Dropout_Max_Entropy_Valid[:,10],color="black", linewidth=2.0,  linestyle='-', label="30th acquisition - Validation")
plt.plot(epochs, Dropout_Max_Entropy_Valid[:,50],color="blue", linewidth=2.0, linestyle='-',label="50th acquisition - Validation")
plt.plot(epochs, Dropout_Max_Entropy_Valid[:,90],color="green", linewidth=2.0, linestyle='-',label="90th acquisition - Validation")


plt.xlabel(r'\textbf{Number of Epochs}')
plt.ylabel(r'\textbf{Accuracy}')
plt.title(r'\textbf{Dropout Max Entropy - Training and Validation Accuracy - Model Fitting}')
plt.grid()
plt.legend(loc = 4)
plt.show()


# plt.figure(figsize=(20, 10), dpi=80)

# plt.figure(1)

# plt.plot(epochs, Dropout_Max_Entropy_Train[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
# plt.plot(epochs, Dropout_Max_Entropy_Train[:,50],color="red", linewidth=1.0, marker='x',label="50th - Training")
# plt.plot(epochs, Dropout_Max_Entropy_Train[:,90],color="red", linewidth=1.0, marker='+',label="90th - Training")

# plt.plot(epochs, Dropout_Max_Entropy_Valid[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
# plt.plot(epochs, Dropout_Max_Entropy_Valid[:,50],color="black", linewidth=1.0, marker='x',label="50th - Validation")
# plt.plot(epochs, Dropout_Max_Entropy_Valid[:,90],color="black", linewidth=1.0, marker='+',label="90th - Validation")

# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy')
# plt.title('Dropout MAX ENTROPY - Training and Validation Accuracy')
# plt.grid()
# plt.legend(loc = 4)


# plt.figure(figsize=(20, 10), dpi=80)
# plt.figure(2)
# plt.plot(epochs, Dropout_Max_Entropy_Train, color="red", linewidth=1.0, marker='x')
# plt.plot(epochs, Dropout_Max_Entropy_Valid, color="black", linewidth=1.0, marker='+' )
# plt.plot(epochs, Dropout_Max_Entropy_Train[:, 50], color="red", linewidth=1.0, marker='x', label="Training Accuracy")
# plt.plot(epochs, Dropout_Max_Entropy_Valid[:, 50], color="black", linewidth=1.0, marker='+', label="Validation Accuracy" )

# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy')
# plt.title('Dropout MAX ENTROPY- Training and Validation Accuracy')
# plt.grid()
# plt.legend(loc = 4)


# plt.show()