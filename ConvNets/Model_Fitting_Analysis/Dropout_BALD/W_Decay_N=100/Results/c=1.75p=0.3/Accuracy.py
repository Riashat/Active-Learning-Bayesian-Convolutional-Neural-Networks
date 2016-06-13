# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, rcParams
import matplotlib.dates as dates


Bald = np.load('BALD.npy')


Queries = np.arange(0, 85, 5)

plt.figure(figsize=(12, 8), dpi=80)

plt.plot(Queries, Bald, color="red", linewidth=3.0, marker='x', label="Dropout BALD" )


plt.xlabel('Number of Queries')
plt.ylabel('Accuracy on Test Set')
plt.title('Dropout BALD - 5 Queries per Acquisition, 100 Labelled Samples')
plt.grid()
# Set x limits
# plt.xlim(1000.0, 10000.0)
# plt.ylim(0.8, 1.0)
plt.legend(loc = 4)
plt.show()



  # plot(np.arange(100), tmpData, label=r'\textbf{Line 1}', linewidth=2)

  # ylabel(r'\textbf{Y-AXIS}', fontsize=20)
  # xlabel(r'\textbf{X-AXIS}', fontsize=20)