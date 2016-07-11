# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
# https://sites.google.com/a/aims-senegal.org/scipy/plotting-with-pylab


import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, rcParams
import matplotlib.dates as dates
import pylab as pl

rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

VarRatio = np.load('Accuracy_Q5_N1000.npy')
Train = np.load('Train_Acc_Q5_N1000.npy')
Valid = np.load('Valid_Acc_Q5_N1000.npy')

Queries = np.arange(100, 1005, 5)
epochs = np.arange(1, 51, 1)


plt.figure(figsize=(15, 10), dpi=80)
# pl.figure() # make separate figure

pl.subplot(2, 1, 1)
plt.plot(Queries, VarRatio, color="blue", linewidth=2.0, marker='^', label=r"\textbf{Dropout Variation Ratio Acquisition Function}" )
plt.xlabel(r'\textbf{Number of Queries}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Dropout Variation Ratio Query Point Acquisition, upto 1000 Labelled Samples}')
plt.grid()
plt.legend(loc = 4)


pl.subplot(2, 1, 2)
plt.plot(epochs, Train[:,10], color="red", linewidth=2.0, marker='.',label=r"\textbf{10th acquisition - training}")
plt.plot(epochs, Train[:,100],color="red", linewidth=2.0, marker='>',label=r"\textbf{100th acquisition - training}")
plt.plot(epochs, Train[:,180],color="red", linewidth=2.0, marker='o',label=r"\textbf{180th acquisition - training}")

plt.plot(epochs, Valid[:,10], color="black", linewidth=2.0, marker='.',label=r"\textbf{10th acquisition - validation}")
plt.plot(epochs, Valid[:,100],color="black", linewidth=2.0, marker='>',label=r"\textbf{100th acquisition - validation}")
plt.plot(epochs, Valid[:,180],color="black", linewidth=2.0, marker='o',label=r"\textbf{180th acquisition - validation}")


plt.xlabel(r'\textbf{Number of Epochs}')
plt.ylabel(r'\textbf{Accuracy}')
plt.title(r'\textbf{Dropout Variation Ratio Model Fitting with Query Point Acquisition}')
plt.grid()
plt.legend(loc = 4)

plt.show()
