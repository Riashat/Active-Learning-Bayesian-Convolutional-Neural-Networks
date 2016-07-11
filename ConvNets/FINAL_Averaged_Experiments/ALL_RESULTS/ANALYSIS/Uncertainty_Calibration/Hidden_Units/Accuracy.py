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


H_1000 = np.load('H_1000.npy')
H_2000 = np.load('H_2000.npy')
H_256 = np.load('H_256.npy')
H_512 = np.load('H_512.npy')




Queries = np.arange(100, 1010, 10)


plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, H_1000, color="blue", linewidth=3.0, marker='x', label=r"\textbf{Size of Hidden Units = 1000}" )
plt.plot(Queries, H_2000, color="green", linewidth=3.0, marker='x', label=r"\textbf{Size of Hidden Units = 2000}" )
plt.plot(Queries, H_512, color="red", linewidth=3.0, marker='x', label=r"\textbf{Size of Hidden Units = 512}" )
plt.plot(Queries, H_256, color="black", linewidth=3.0, marker='x', label=r"\textbf{Size of Hidden Units = 256}" )


plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Dropout BALD: Different Sizes Number of Hidden Units in Top NN layer in CNNs}')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 4)
plt.show()





