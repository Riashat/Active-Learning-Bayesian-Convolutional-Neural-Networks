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




Bald = np.load('Bald.npy')
VarRatio = np.load('VarRatio.npy')
Max_Entropy = np.load('Binary_Max_Entropy.npy')
MBR = np.load('mbr.npy')
Random = np.load('Binary_Random.npy')


Queries = np.arange(100, 510, 10)



plt.figure(figsize=(15, 10), dpi=80)

plt.plot(Queries, Bald[0:41], color="red", linewidth=3.0, marker='x', label=r"\textbf{Dropout BALD}" )
plt.plot(Queries, VarRatio[0:41], color="blue", linewidth=3.0, marker='x', label=r"\textbf{Dropout Variation Ratio}" )
plt.plot(Queries, Max_Entropy[0:41], color="green", linewidth=3.0, marker='x', label=r"\textbf{Dropout Max Entropy}" )
plt.plot(Queries, MBR[0:41], color="black", linewidth=3.0, marker='x', label=r"\textbf{Minimum Bayes Risk}" )
plt.plot(Queries, Random[0:41], color="magenta", linewidth=3.0, marker='x', label=r"\textbf{Random Acquisition}" )


plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Binary Classsification - Comparing Active Learning vs Semi-Supervised Learning}')
plt.grid()
plt.ylim(0.7, 1)
plt.legend(loc = 4)
plt.show()
