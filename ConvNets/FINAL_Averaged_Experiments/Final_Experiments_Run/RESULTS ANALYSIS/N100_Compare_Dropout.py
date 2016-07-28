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




Dropout_Bald = np.load('N100_Dropout_Bald.npy')
Dropout_LC = np.load('N100_Dropout_LC.npy')
Dropout_MaxEntropy = np.load('N100_Dropout_Max_Entropy.npy')
Segnet = np.load('N100_Segnet.npy')
VarRatio = np.load('N100_VarRatio.npy')

Random = np.load('N100_Random.npy')
BvSB = np.load('N100_BvSB.npy')


Q1 = np.arange(20, 101, 1)

# Q10 = np.arange(100, 1010, 10)



plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Q1, Dropout_Bald, color="red", linewidth=2.0, marker='.', label=r"\textbf{Dropout BALD}" )
plt.plot(Q1, Dropout_LC, color="black", linewidth=2.0, marker='.', label=r"\textbf{Dropout Least Confident}" )
plt.plot(Q1, Dropout_MaxEntropy, color="blue", linewidth=2.0, marker='.', label=r"\textbf{Dropout Max Entropy}" )
plt.plot(Q1, VarRatio, color="green", linewidth=2.0, marker='.', label=r"\textbf{Dropout Variation Ratio}" )
plt.plot(Q1, Segnet, color="magenta", linewidth=2.0, marker='.', label=r"\textbf{Dropout Bayes Segnet}" )
plt.plot(Q1, Random, color="cyan", linewidth=2.0, marker='.', label=r"\textbf{Random}" )
plt.plot(Q1, BvSB, color="yellow", linewidth=2.0, marker='.', label=r"\textbf{Best vs Second Best}" )


plt.xlabel(r'\textbf{Number of Labelled Samples from Pool Set}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Active Learning Acquisition Functions - 100 Labelled Samples for Training on MNIST}')
plt.grid()

plt.legend(loc = 4)
plt.show()
