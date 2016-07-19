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


DGP_Bald = np.load('DGPEP_Bald.npy')
PBP_Random = np.load('PBP_Random.npy')
BB_Alpha_0p5 = np.load('BB_Alpha0p5_Random.npy')
PBP_Bald = np.load('PBP_Bald.npy')


Q = np.arange(20, 401, 1)




plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Q, DGP_Bald, color="red", linewidth=1.0, marker='x', label=r"\textbf{AEP DGP Bald Acquisition}" )
plt.plot(Q, PBP_Random, color="black", linewidth=1.0, marker='x', label=r"\textbf{PBP Random Acquisition}" )
plt.plot(Q, PBP_Bald, color="blue", linewidth=1.0, marker='x', label=r"\textbf{PBP Bald Acquisition}" )
plt.plot(Q, BB_Alpha_0p5, color="green", linewidth=1.0, marker='x', label=r"\textbf{Black-Box Alpha=0.5 Random Acquisition}" )



plt.xlabel(r'\textbf{Number of Samples from Pool Set}')
plt.ylabel(r'\textbf{Accuracy Test RMSE}')
plt.title(r'\textbf{Active Learning in Boston Housing }')
plt.grid()
# Set x limits
# plt.xlim(0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 1)
plt.show()
