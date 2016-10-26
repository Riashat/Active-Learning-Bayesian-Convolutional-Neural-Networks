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
BvSB = np.load('BvSB.npy')
Dropout_Max_Entropy = np.load('Dropout_Max_Entropy.npy')
Segnet = np.load('Dropout_Segnet.npy')
VarRatio = np.load('Dropout_VarRatio.npy')
Max_Entropy = np.load('Max_Entropy.npy')
Random = np.load('Random.npy')




Queries = np.arange(20, 1010, 10)
# Q10 = np.arange(100, 1010, 10)



plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, Dropout_Bald, color="red", linewidth=3.0, marker='x', label=r"\textbf{Dropout BALD}" )
plt.plot(Queries, Dropout_Max_Entropy, color="black", linewidth=3.0, marker='x', label=r"\textbf{Dropout Max Entropy}" )
plt.plot(Queries, VarRatio, color="blue", linewidth=3.0, marker='x', label=r"\textbf{Dropout Variation Ratio}" )
plt.plot(Queries, Segnet, color="magenta", linewidth=3.0, marker='x', label=r"\textbf{Dropout Bayes Segnet}" )


plt.plot(Queries, Random, color="orange", linewidth=3.0, marker='x', label=r"\textbf{Random}" )
plt.plot(Queries, BvSB, color="yellow", linewidth=3.0, marker='x', label=r"\textbf{Best vs Second Best}" )
plt.plot(Queries, Max_Entropy, color="green", linewidth=3.0, marker='x', label=r"\textbf{Max Entropy}" )



plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Acquisition Functions - 10 Queries per Acquisition, 1000 Labelled Samples}')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 4)
plt.show()
