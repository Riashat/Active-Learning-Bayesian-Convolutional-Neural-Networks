# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, rcParams
import matplotlib.dates as dates
import pylab as pl

  # activate latex text rendering
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']




Dropout_Bald = np.load('Dropout_Bald.npy')
Dropout_Max_Entropy = np.load('Dropout_Max_Entropy.npy')
Dropout_Segnet = np.load('Dropout_Segnet.npy')
Dropout_VarRatio = np.load('Dropout_VarRatio.npy')

Softmax_Max_Entropy = np.load('Softmax_Max_Entropy.npy')
Softmax_Segnet = np.load('Softmax_Segnet.npy')
Softmax_VarRatio = np.load('Softmax_VarRatio.npy')
Softmax_Bald = np.load('Softmax_Bald.npy')


Queries = np.arange(20, 1010, 10)



plt.figure(figsize=(15, 10), dpi=80)

pl.subplot(2, 2, 1) 
plt.plot(Queries, Dropout_Bald, color="red", linewidth=1.0, marker='x', label=r"\textbf{Dropout BALD}" )
plt.plot(Queries, Softmax_Bald, color="black", linewidth=1.0, marker='.', label=r"\textbf{Softmax BALD}" )

plt.title(r'\textbf{BALD}')
plt.grid()
plt.legend(loc = 4)


pl.subplot(2, 2, 2)
plt.plot(Queries, Dropout_Max_Entropy, color="red", linewidth=1.0, marker='x', label=r"\textbf{Dropout Max Entropy}" )
plt.plot(Queries, Softmax_Max_Entropy, color="black", linewidth=1.0, marker='.', label=r"\textbf{Softmax Max Entropy}" )
plt.title(r'\textbf{Max Entropy}')
plt.grid()
plt.legend(loc = 4)

pl.subplot(2, 2, 3)
plt.plot(Queries, Dropout_VarRatio, color="red", linewidth=1.0, marker='x', label=r"\textbf{Dropout Variation Ratio}" )
plt.plot(Queries, Softmax_VarRatio, color="black", linewidth=1.0, marker='.', label=r"\textbf{Softmax Variation Ratio}" )
plt.title(r'\textbf{Variation Ratio}')
plt.grid()
plt.legend(loc = 4)

pl.subplot(2, 2, 4)
plt.plot(Queries, Dropout_Segnet, color="red", linewidth=1.0, marker='x', label=r"\textbf{Dropout Bayes Segnet}" )
plt.plot(Queries, Softmax_Segnet, color="black", linewidth=1.0, marker='.', label=r"\textbf{Softmax Bayes Segnet}" )
plt.title(r'\textbf{Bayes Segnet}')
plt.grid()
plt.legend(loc = 4)


plt.show()
