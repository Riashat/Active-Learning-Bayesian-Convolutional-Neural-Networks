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


Bald = np.load('Bald_Average.npy')
BvSB = np.load('BvSB_Average.npy')
Dropout_ME = np.load('Dropout_ME_Average.npy')
Max_Entropy = np.load('Max_Entropy_Average.npy')
Random = np.load('Random_Average.npy')
VarRatio = np.load('VarRatio_Average.npy')


Queries = np.arange(20, 101, 1)



plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, Bald, color="red", linewidth=3.0, marker='x', label=r"\textbf{Dropout BALD}" )
plt.plot(Queries, BvSB, color="yellow", linewidth=2.0, marker='x', label=r"\textbf{Best vs Second Best}" )
plt.plot(Queries, Dropout_ME, color="black", linewidth=3.0, marker='x', label=r"\textbf{Dropout Max Entropy}" )
plt.plot(Queries, Max_Entropy, color="green", linewidth=2.0, marker='x', label=r"\textbf{Max Entropy}" )
plt.plot(Queries, Random, color="orange", linewidth=2.0, marker='x', label=r"\textbf{Random}" )
plt.plot(Queries, VarRatio, color="blue", linewidth=3.0, marker='x', label=r"\textbf{Dropout Variation Ratio}" )



plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Acquisition Functions - 1 Queries per Acquisition, 100 Labelled Samples}')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 4)
plt.show()
