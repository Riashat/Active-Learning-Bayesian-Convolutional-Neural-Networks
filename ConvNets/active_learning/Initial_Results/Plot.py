# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

# Create a figure of size 8x6 inches, 80 dots per inch
plt.figure(figsize=(8, 6), dpi=80)


Random_Acc = np.array([0.959, 0.965, 0.964, 0.971, 0.970, 0.973, 0.973, 0.971, 0.961, 0.975])

Max_Entropy_Acc = np.array([0.959, 0.974, 0.961, 0.956, 0.941, 0.953, 0.931, 0.934, 0.910, 0.884])

Number_Queries = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,  ])


Queries = np.array([1000, 2000])

Dropout_BALD = np.array([0.923, 0.926])

Only_Entropy = np.array([0.919, 0.908])


plt.plot(Number_Queries, Random_Acc, color="blue", linewidth=1.0, marker = 'o', label="Random Acquisitions")


plt.plot(Number_Queries, Max_Entropy_Acc, color="red", linewidth=1.0, marker = 'o', label="Max Entropy Based Acquisitions")

# plt.plot(Queries, Dropout_BALD, color="green", linewidth=1.0, marker = 'o', label="Dropout BALD Acquisition Function")


# plt.plot(Queries, Only_Entropy, color="black", linewidth=1.0, marker = 'o', label="Only Entropy Acquisition Function")




plt.xlabel('Number of Queries')
plt.ylabel('Accuracy on Test Set')
plt.title('Comparing Acquisition Functions')
plt.grid()
# Set x limits
plt.xlim(1000.0, 10000.0)


plt.ylim(0.8, 1.0)
plt.legend(loc = 4)


plt.show()