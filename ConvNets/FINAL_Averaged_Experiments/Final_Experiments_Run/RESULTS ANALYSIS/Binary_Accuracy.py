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




Bald = np.load('Binary_Bald.npy')
MBR = np.load('Binary_MBR.npy')
Max_Entropy = np.load('Binary_Max_Entropy.npy')
Random = np.load('Binary_Random.npy')
VarRatio = np.load('Binary_VarRatio.npy')



Queries = np.arange(100, 610, 10)



plt.figure(figsize=(15, 10), dpi=80)

plt.plot(Queries, Bald, color="red", linewidth=3.0, marker='^', label=r"\textbf{Dropout BALD}" )
plt.plot(Queries, VarRatio, color="blue", linewidth=3.0, marker='o', label=r"\textbf{Dropout Variation Ratio}" )
plt.plot(Queries, Max_Entropy[0:51], color="green", linewidth=3.0, marker='.', label=r"\textbf{Dropout Max Entropy}" )
plt.plot(Queries, MBR, color="black", linewidth=3.0, marker='x', label=r"\textbf{Minimum Bayes Risk (MBR)}" )
# plt.plot(Queries, Random[0:51], color="magenta", linewidth=2.0, marker='x', label=r"\textbf{Random Acquisition}" )


plt.xlabel(r'\textbf{Number of Samples from the Pool Set}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Binary Classsification - Comparing Active Learning and Semi-Supervised Learning}')
plt.grid()
plt.ylim(0.5, 1)
plt.legend(loc = 4)
plt.show()
