# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, rcParams
import matplotlib.dates as dates
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']



N = np.array([100, 600, 1000, 1700, 3000])

Bald = np.array([0.8326, 0.9528, 0.9803, 0.9812, 0.9830])

Dropout_Max_Entropy = np.array([0.8363, 0.9817, 0.9780, 0.9902, 0.9818])

Variation_Ratio = np.array([0.8055, 0.9801, 0.9791, 0.9840, 0.9892])

Segnet = np.array([0.8061, 0.8692, 0.8821, 0.9257, 0.9147])

plt.figure(figsize=(12, 8), dpi=80)


plt.plot(N, Bald, color="red", linewidth=3.0, marker='o', label=r"\textbf{Dropout BALD}" )
plt.plot(N, Dropout_Max_Entropy, color="black", linewidth=3.0, marker='o', label=r"\textbf{Dropout Max Entropy}")
plt.plot(N, Variation_Ratio, color="orange", linewidth=3.0, marker='o', label=r"\textbf{Dropout Variation Ratio}")
plt.plot(N, Segnet, color="blue",linewidth=3.0, marker='o', label=r"\textbf{Dropout Bayes Segnet}" )



plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Accuracy of Acquisition Functions vs Number of Labelled Samples for Training after Pooling}')
plt.grid()
# Set x limits
plt.xlim(0.0, 3500.0)
#plt.ylim(0, 3500.0)
plt.legend(loc = 4)
plt.show()

