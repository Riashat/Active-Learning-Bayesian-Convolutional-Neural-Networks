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


Kernel_1 = np.load('Kernel_1.npy')
Kernel_2 = np.load('Kernel_2.npy')
Kernel_4 = np.load('Kernel_4.npy')
Kernel_5 = np.load('Kernel_5.npy')


Queries = np.arange(100, 1010, 10)


plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, Kernel_1, color="blue", linewidth=3.0, marker='x', label=r"\textbf{Size of Gaussian Filter = 1}" )
plt.plot(Queries, Kernel_2, color="green", linewidth=3.0, marker='x', label=r"\textbf{Size of Gaussian Filter = 2}" )
plt.plot(Queries, Kernel_4, color="red", linewidth=3.0, marker='x', label=r"\textbf{Size of Gaussian Filter = 4}" )
plt.plot(Queries, Kernel_5, color="black", linewidth=3.0, marker='x', label=r"\textbf{Size of Gaussian Filter = 5}" )


plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Different Sizes of Gaussian Filter Kernel in CNNs}')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 4)
plt.show()





