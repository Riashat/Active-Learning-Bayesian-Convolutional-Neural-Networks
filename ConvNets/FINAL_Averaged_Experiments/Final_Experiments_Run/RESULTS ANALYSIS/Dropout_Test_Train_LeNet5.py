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


Dropout_All_Q10 = np.load('LeNet5_All')
Dropout_All_Q5 = np.load('LeNet5_All2.npy')

Dropout_Bald = np.load('Dropout_Bald.npy')

Dropout_OnlyTest = np.load('LeNet5_Only_TestTime.npy')
Dropout_OnlyTrain = np.load('LeNet5_Only_Training.npy')


Queries = np.arange(100, 1010, 10)
# Q10 = np.arange(100, 1010, 10)


plt.figure(figsize=(12, 8), dpi=80)
plt.plot(Queries, Dropout_Bald[8:99], color="red", linewidth=3.0, marker='^', label=r"\textbf{Dropout Both Training and Test TIme}" )
plt.plot(Queries, Dropout_OnlyTest, color="black", linewidth=3.0, marker='.', label=r"\textbf{Dropout Only Test Time" )
plt.plot(Queries, Dropout_OnlyTrain, color="blue", linewidth=3.0, marker='x', label=r"\textbf{Dropout Only Training}" )



plt.xlabel(r'\textbf{Number of Labelled Samples from Pool Set}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Comparing Dropout BALD Acquisition - Model Architectures}')
plt.grid()

plt.legend(loc = 4)
plt.show()
