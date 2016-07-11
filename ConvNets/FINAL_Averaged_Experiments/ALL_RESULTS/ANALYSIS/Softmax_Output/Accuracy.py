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


Max_Entropy = np.load('Max_Entropy.npy')
Segnet = np.load('Segnet.npy')
VarRatio = np.load('VarRatio.npy')



Queries = np.arange(100, 3010, 10)



plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, Max_Entropy, color="red", linewidth=3.0, marker='x', label=r"\textbf{Dropout Max Entropy}" )
plt.plot(Queries, Segnet, color="blue", linewidth=3.0, marker='x', label=r"\textbf{Dropout Bayes Segnet}" )
plt.plot(Queries, VarRatio, color="black", linewidth=3.0, marker='x', label=r"\textbf{Dropout Variation Ratio}" )



plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Different Non-Linearity or Activation Functions in BCNN}')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 4)
plt.show()
