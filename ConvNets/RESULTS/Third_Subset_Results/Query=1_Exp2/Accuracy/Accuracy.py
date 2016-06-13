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


Bald = np.load('BALD.npy')
Bayes_Segnet = np.load('Segnet.npy')
BCNN_Entropy = np.load('Dropout_Max_Entropy.npy')
Variation = np.load('Variation_Ratio.npy')


Queries = np.arange(0, 81, 1)

plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, Bald, color="red", linewidth=3.0, marker='x', label=r"\textbf{Dropout BALD}" )
plt.plot(Queries, Bayes_Segnet, color="blue", linewidth=3.0, marker='x', label=r"\textbf{Dropout Bayes Segnet}")
plt.plot(Queries, BCNN_Entropy, color="black", linewidth=3.0, marker='x', label=r"\textbf{Dropout - Max Entropy}")
plt.plot(Queries, Variation, color="orange",linewidth=3.0, marker='x', label=r"\textbf{Dropout - Variation Ratio}" )


plt.xlabel(r'\textbf{Number of Queries}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Comparing Acquisition Functions - 1 Query per Acquisition, 100 Labelled Samples}')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 3)
plt.show()



  # plot(np.arange(100), tmpData, label=r'\textbf{Line 1}', linewidth=2)

  # ylabel(r'\textbf{Y-AXIS}', fontsize=20)
  # xlabel(r'\textbf{X-AXIS}', fontsize=20)