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
Dropout_Max_Entropy = np.load('Dropout_Max_Entropy.npy')
Dropout_Segnet = np.load('Dropout_Segnet.npy')
Dropout_VarRatio = np.load('Dropout_VarRatio.npy')

Softmax_Max_Entropy = np.load('Softmax_Max_Entropy.npy')
Softmax_Segnet = np.load('Softmax_Segnet.npy')
Softmax_VarRatio = np.load('Softmax_VarRatio.npy')
Softmax_Bald = np.load('Softmax_Bald.npy')

Random = np.load('Random.npy')
BvSB = np.load('BvSB.npy')

Queries = np.arange(20, 1010, 10)
# Q10 = np.arange(100, 1010, 10)



plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, Dropout_Bald, color="red", linewidth=1.0, marker='.', label=r"\textbf{Dropout BALD}" )
plt.plot(Queries, Dropout_Max_Entropy, color="black", linewidth=1.0, marker='.', label=r"\textbf{Dropout Max Entropy}" )
plt.plot(Queries, Dropout_Segnet, color="blue", linewidth=1.0, marker='.', label=r"\textbf{Dropout Bayes Segnet}" )
plt.plot(Queries, Dropout_VarRatio, color="green", linewidth=1.0, marker='.', label=r"\textbf{Dropout Variation Ratio}" )

plt.plot(Queries, Softmax_Bald, color="red", linewidth=1.0, marker='x', label=r"\textbf{Softmax BALD}" )
plt.plot(Queries, Softmax_Segnet, color="black", linewidth=1.0, marker='x', label=r"\textbf{Softmax Segnet}" )
plt.plot(Queries, Softmax_VarRatio, color="blue", linewidth=1.0, marker='x', label=r"\textbf{Softmax Variation Ratio}" )
plt.plot(Queries, Softmax_Max_Entropy, color="green", linewidth=1.0, marker='x', label=r"\textbf{Softmax Max Entropy}" )

plt.plot(Queries, Random, color="cyan", linewidth=1.0, marker='^', label=r"\textbf{Random}" )
plt.plot(Queries, BvSB, color="magenta", linewidth=1.0, marker='^', label=r"\textbf{Best vs Second Best}" )



plt.xlabel(r'\textbf{Number of Labelled Samples from Pool Set}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Comparing Active Learning Acquisition Functions - 10 Queries, 1000 Labelled Samples on MNIST}')
plt.grid()

plt.legend(loc = 4)
plt.show()
