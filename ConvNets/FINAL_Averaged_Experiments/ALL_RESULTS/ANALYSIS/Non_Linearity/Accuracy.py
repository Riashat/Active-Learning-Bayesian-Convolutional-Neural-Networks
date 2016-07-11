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


Linear = np.load('Linear.npy')
ReLU = np.load('ReLU.npy')
Sigmoid = np.load('Sigmoid.npy')
TanH = np.load('TanH.npy')


Queries = np.arange(100, 1010, 10)



plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, Linear, color="red", linewidth=3.0, marker='x', label=r"\textbf{Linear}" )
plt.plot(Queries, ReLU, color="blue", linewidth=3.0, marker='x', label=r"\textbf{ReLU}" )
plt.plot(Queries, Sigmoid, color="black", linewidth=3.0, marker='x', label=r"\textbf{Sigmoid}" )
plt.plot(Queries, TanH, color="green", linewidth=3.0, marker='x', label=r"\textbf{TanH}" )



plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Signnificance of Non-Linearity in Bayesian ConvNet}')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 4)
plt.show()
