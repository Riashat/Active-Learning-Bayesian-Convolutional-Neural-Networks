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


# PLOT WITH 1000 FINAL QUERIES AT THE END

Bald_100 = np.load('100_BALD.npy')
Bald_10 = np.load('10_BALD.npy')
#Bald_1 = np.load('1_BALD.npy')
Bald_5 = np.load('5_BALD.npy')

Q_10 = np.arange(0, 960, 10)
Q_100 = np.arange(0, 1050, 100)
Q_1 = np.arange(0, 51, 1)
Q_5 = np.arange(0, 955, 5)

plt.figure(figsize=(15, 10), dpi=80)
plt.plot(Q_5, Bald_5, color="red", linewidth=2.0, marker='o', label=r"\textbf{5 Queries per Acquisition}")
plt.plot(Q_10, Bald_10, color="blue", linewidth=2.0, marker='o', label=r"\textbf{10 Queries per Acquisition}")
plt.plot(Q_100, Bald_100, color="black", linewidth=2.0, marker='o', label=r"\textbf{100 Queries per Acquisition}")
#plt.plot(Q_1, Bald_1, color="green", linewidth=1.0, marker='o', label="1 Query per Acquisition")

plt.xlabel(r'\textbf{Number of Queries}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Significance of Queries Per Acquisition - Dropout BALD}')
plt.grid()
plt.legend(loc = 4)
plt.show()



# PLOT WITH 100 FINAL QUERIES AT THE END

# Bald_100 = np.load('100_BALD.npy')
# Bald_10 = np.load('10_BALD.npy')
# Bald_10 = Bald_10[0:6]

# Bald_1 = np.load('1_BALD.npy')

# Q_10 = np.arange(50, 110, 10)
# # Q_100 = np.arange(0, 110, 100)
# Q_1 = np.arange(50, 101, 1)

# plt.figure(figsize=(15, 10), dpi=80)

# plt.plot(Q_10, Bald_10, color="red", linewidth=1.0, marker='o', label=r"\textbf{10 Queries per Acquisition}")
# # plt.plot(Q_100, Bald_100, color="black", linewidth=1.0, marker='o', label="100 Queries per Acquisition")
# plt.plot(Q_1, Bald_1, color="green", linewidth=1.0, marker='o', label=r"\textbf{1 Query per Acquisition}")

# plt.xlabel(r'\textbf{Number of Queries}')
# plt.ylabel(r'\textbf{Accuracy on Test Set}')
# plt.title(r'\textbf{Significance of Queries Per Acquisition - Dropout BALD}')
# plt.grid()
# plt.legend(loc = 4)

# plt.show()





