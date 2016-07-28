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


Dropout_Bald_TanH_0p005 = np.load('Dropout_Bald_TanH_0p005.npy')
Dropout_Bald_V2 = np.load('Dropout_Bald_trial2.npy')
Dropout_Bald_V3 = np.load('Dropout_Bald_trial3.npy')
Dropout_Random_V3 = np.load('Dropout_Random_trial3.npy')

BB_Alpha_Random_0p5 = np.load('BBAlpha_0p5_Random.npy')
BB_Alpha_Random_10m6 = np.load('BBAlpha_10em6_Random.npy')
BB_Alpha_Random_1 = np.load('BBAlpha_1_Random.npy')


PBP_Bald = np.load('PBP_Bald.npy')
PBP_Random = np.load('PBP_Random.npy')


Dropout_Bald_YarinConfigs = np.load('Dropout_Bald_YarinConfigs.npy')
Dropout_Bald_TanH_0p05 = np.load('Dropout_Bald_NewConfigsV4.npy')





Q = np.arange(20, 401, 1)


plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Q, Dropout_Bald_TanH_0p005, color="red", linewidth=1.0, marker='^', label=r"\textbf{Dropout TanH p=0.005 BALD}" )
plt.plot(Q, Dropout_Bald_YarinConfigs, color="black", linewidth=1.0, marker='^', label=r"\textbf{Dropout Yarin Config}" )
plt.plot(Q, Dropout_Bald_TanH_0p05, color="blue", linewidth=1.0, marker='^', label=r"\textbf{Dropout TanH p=0.05 BALD}" )




plt.xlabel(r'\textbf{Number of Samples from Pool Set}')
plt.ylabel(r'\textbf{Accuracy Test RMSE}')
plt.title(r'\textbf{Active Learning in Boston Housing }')
plt.grid()
# Set x limits
# plt.xlim(0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 1)
plt.show()
