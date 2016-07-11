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


GoogLeNet = np.load('GoogLeNet.npy')
LeNet5_All = np.load('LeNet5_All.npy')
LeNet5_Inner = np.load('LeNet5_Inner.npy')
LeNet5_All2 = np.load('LeNet5_All2.npy')
LeNet5_Inner_Q5 = np.load('LeNet5_Inner_Q5_N1000.npy')

Queries = np.arange(100, 1010, 10)
Queries2 = np.arange(100, 1005, 5)



plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, GoogLeNet, color="red", linewidth=3.0, marker='x', label=r"\textbf{GoogLeNet architecture}" )
plt.plot(Queries, LeNet5_All[0:91], color="blue", linewidth=3.0, marker='x', label=r"\textbf{LeNet5 - Dropout after All Layers and Test-Time Dropout}" )
plt.plot(Queries, LeNet5_Inner, color="black", linewidth=3.0, marker='x', label=r"\textbf{LeNet5 - Dropout Only Test-Time}" )
# plt.plot(Queries2, LeNet5_All2, color="green", linewidth=3.0, marker='x', label=r"\textbf{LeNet5 - Dropout All Layers Q=5}" )
# plt.plot(Queries2, LeNet5_Inner_Q5, color="purple", linewidth=3.0, marker='x', label=r"\textbf{LeNet5 - Dropout Inner Layers Q=5}" )



plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Comparing different model architectures}')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 4)
plt.show()
