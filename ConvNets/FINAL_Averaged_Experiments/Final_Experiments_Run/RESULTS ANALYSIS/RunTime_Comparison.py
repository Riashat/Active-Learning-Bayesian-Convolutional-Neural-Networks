# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, rcParams
import matplotlib.dates as dates
import matplotlib as mpl


# activate latex text rendering
rc('text', usetex=True)
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

Bald_Q5 = np.load('RunTime_Bald_Q5_Accuracy.npy')
Bald_Q10 = np.load('RunTime_Bald_Q10_Accuracy.npy')
Bald_Q100 = np.load('RunTime_Bald_Q100_Accuracy.npy')

Q5_Time = np.load('RunTime_Bald_Q5_Time.npy')
Q10_Time = np.load('Runtime_Bald_Q10_Time.npy')
Q100_Time = np.load('RunTime_Bald_Q100_Time.npy')


Bald_Q1_v2 = np.load('Bald_Q1_N1000.npy')
Bald_Q5_v2 = np.load('Bald_Q5_N1000.npy')
Bald_Q10_N3000_v2 = np.load('Bald_Q10_N3000.npy')

col_labels=['Total Run Time (Hours)']
row_labels=['Query = 5','Query = 10','Query = 100']
table_vals=[[31.9],[16.78],[2.04]]


Q5= np.arange(20, 1005, 5)
Q10 = np.arange(20, 1010, 10)
Q100 = np.arange(100, 1100, 100)

# Q10 = np.arange(100, 1010, 10)

Q1_v2 = np.arange(100, 1001, 1)
# Q5_v2 = np.arange(100, 1005, 5)
# Q10_v2 = np.arange(100, 1010, 10)



plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Q5, Bald_Q5, color="red", linewidth=1.0, marker='^', label=r"\textbf{Dropout BALD Queries = 5}" )
plt.plot(Q10, Bald_Q10, color="black", linewidth=1.0, marker='o', label=r"\textbf{Dropout BALD Queries = 10}" )
plt.plot(Q100, Bald_Q100, color="blue", linewidth=1.0, marker='.', label=r"\textbf{Dropout BALD Queries = 100}" )

#plt.plot(Q1_v2, Bald_Q1_v2, color="green", linewidth=1.0, marker='x', label=r"\textbf{Dropout BALD Queries = 1 V2}" )
#plt.plot(Q5_v2, Bald_Q5_v2, color="black", linewidth=1.0, marker='^', label=r"\textbf{Dropout BALD Queries = 5 V2}" )
# plt.plot(Q10_v2, Bald_Q10_N3000_v2[0:91], color="blue", linewidth=1.0, marker='^', label=r"\textbf{Dropout BALD Queries = 10 V2}" )


plt.xlabel(r'\textbf{Number of Labelled Samples from Pool Set}')
plt.ylabel(r'\textbf{Accuracy on Test Set}')
plt.title(r'\textbf{Comparing Query Rate using Dropout BALD}')
plt.grid()

plt.legend(loc = 4)
# the rectangle is where I want to place the table
the_table = plt.table(cellText=table_vals,
                  colWidths = [0.3]*3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='center right')
plt.text(12,3.4,'Table Title',size=20)


plt.show()
