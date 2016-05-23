# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

# Create a figure of size 8x6 inches, 80 dots per inch
plt.figure(figsize=(8, 6), dpi=80)


Data_Size = np.array([500, 3500, 4500, 7500, 11000])
Acc = np.array([0.09, 0.153, 0.19, 0.322, 0.392])

plt.plot(Data_Size, Acc, color="blue", linewidth=1.0, marker = 'o', label="Manual Acquisition of Training Set")

X = np.array([1,2,3,4,5,6,7,8,9,10])


plt.xlabel('Training Data Size')
plt.ylabel('Accuracy of CNN Classifier')
plt.title('Effect of Increasing Training Sete Size')
plt.grid()
# Set x limits
plt.xlim(400.0, 12000.0)


plt.ylim(0, 0.4)
plt.legend(loc = 4)


plt.show()