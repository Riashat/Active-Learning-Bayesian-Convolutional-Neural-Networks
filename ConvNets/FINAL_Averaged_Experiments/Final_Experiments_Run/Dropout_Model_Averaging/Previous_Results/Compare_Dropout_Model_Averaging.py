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


Dropout_Bald = np.load('Dropout_Bald.npy')
Dropout_Model_Averaging_Bald = np.load('Dropout_Model_Averaging_Bald.npy')
Dropout_Model_Averaging_LC = np.load('Dropout_Model_Averaging_LC.npy')





Q_100 = np.arange(100, 1010, 10)
Q_20 = np.arange(20, 1010, 10)


plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Q_20, Dropout_Bald, color="red", linewidth=1.0, marker='.', label=r"\textbf{Dropout BALD}" )
plt.plot(Q_100, Dropout_Model_Averaging_Bald, color="black", linewidth=1.0, marker='.', label=r"\textbf{Dropout + Model Averaging BALD}" )
plt.plot(Q_100, Dropout_Model_Averaging_LC, color="green", linewidth=1.0, marker='.', label=r"\textbf{Dropout + Model Averaging Least Confidence}" )


plt.xlabel(r'\textbf{Number of Labelled Samples from Pool Set}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Dropout and Dropout + Model Averaging}')
plt.grid()

plt.legend(loc = 4)
plt.show()
