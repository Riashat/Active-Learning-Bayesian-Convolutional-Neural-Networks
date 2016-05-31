# http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 21, 1)

Highest_Entropy_Train_Loss = np.load('Max_Entropy_Train.npy')
Highest_Entropy_Train_Loss = Highest_Entropy_Train_Loss[0:20, 0:31]

Highest_Entropy_Valid_Loss = np.load('Max_Entropy_Valid.npy')
Highest_Entropy_Valid_Loss = Highest_Entropy_Valid_Loss[0:20, 0:31]

plt.figure(figsize=(20, 10), dpi=80)

plt.figure(1)

plt.plot(epochs, Highest_Entropy_Train_Loss[:,1], color="red", linewidth=1.0, marker='o',label="1st - Training")
plt.plot(epochs, Highest_Entropy_Train_Loss[:,10],color="red", linewidth=1.0, marker='x',label="10th - Training")
plt.plot(epochs, Highest_Entropy_Train_Loss[:,20],color="red", linewidth=1.0, marker='+',label="20th - Training")

plt.plot(epochs, Highest_Entropy_Valid_Loss[:,1], color="black", linewidth=1.0, marker='o',label="1st - Validation")
plt.plot(epochs, Highest_Entropy_Valid_Loss[:,10],color="black", linewidth=1.0, marker='x',label="10th - Validation")
plt.plot(epochs, Highest_Entropy_Valid_Loss[:,20],color="black", linewidth=1.0, marker='+',label="20th - Validation")

plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Max Entropy - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.figure(figsize=(20, 10), dpi=80)
plt.figure(2)
plt.plot(epochs, Highest_Entropy_Train_Loss, color="red", linewidth=1.0, marker='x')
plt.plot(epochs, Highest_Entropy_Valid_Loss, color="black", linewidth=1.0, marker='+' )


plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Max Entropy - Training and Validation Loss')
plt.grid()
plt.legend(loc = 1)


plt.show()