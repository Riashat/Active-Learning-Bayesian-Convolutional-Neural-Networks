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




Bald100 = np.load('Bald_100.npy')
Bald1000 = np.load('Bald_1000.npy')

VarRatio100 = np.load('VarRatio_100.npy')
VarRatio1000 = np.load('VarRatio_1000.npy')



Queries100 = np.arange(20, 101, 1)
Queries1000 = np.arange(100, 1005, 5)



# plt.figure(figsize=(12, 8), dpi=80)

# plt.plot(Queries100, Bald100, color="red", linewidth=3.0, marker='x', label=r"\textbf{Dropout Bald - 100 Samples}" )
# plt.plot(Queries100, VarRatio100, color="blue", linewidth=3.0, marker='x', label=r"\textbf{Dropout Variation Ratio - 100 Samples}" )


# plt.xlabel(r'\textbf{Number of Labelled Samples}')
# plt.ylabel(r'\textbf{Accuracy on Test Set}')
# plt.title(r'\textbf{100 Labelled Samples}')
# plt.grid()
# # Set x limits
# # plt.xlim(1000.0, 10000.0)
# # plt.ylim(0.8, 1.0)
# plt.legend(loc = 4)
# plt.show()



plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries1000, Bald1000, color="red", linewidth=3.0, marker='x', label=r"\textbf{Dropout Bald - 1000 Samples}" )
plt.plot(Queries1000, VarRatio1000, color="blue", linewidth=3.0, marker='x', label=r"\textbf{Dropout Variation Ratio - 1000 Samples}" )


plt.xlabel(r'\textbf{Number of Labelled Samples}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{1000 Labelled Samples}')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 4)
plt.show()





