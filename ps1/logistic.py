import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

alpha = 0.01
theta = np.zeros(3)
epsilon = 0.001

def sigmoid(theta, x):
    return 1.0 / (1.0 + math.exp(-np.dot(theta, x)))


xdata = pd.read_csv("logistic_x.txt", sep='\s+', header=None)
ydata = pd.read_csv("logistic_y.txt", sep='\s+', header=None)

x = xdata.values
y = ydata.values
intercept = np.ones((y.size, 1))
x = np.concatenate((intercept, x), axis=1)

i = np.random.randint(y.size)
grad = sigmoid(theta, -y[i,:] * x[i,:]) * y[i,:] * x[i,:]
j = 1
k = 1
while (np.linalg.norm(grad) > epsilon):
    theta += alpha * grad
    i = np.random.randint(y.size)
    grad = sigmoid(theta, -y[i,:] * x[i,:]) * y[i,:] * x[i,:]
    #print(str(k) + "    " + str(theta) + '\n')
    if (j == 20000):
        alpha /= 10
        j = 1
    j += 1
    k += 1
    if (k > 100000): break
    
print(theta)

plt.scatter(x[:, 1], x[:, 2], c=y[:, 0])

plotx = np.arange(np.amin(x[:, 1]), np.amax(x[:, 1]), 0.01)
plt.plot(plotx, -theta[0]/theta[2] - (theta[1]/theta[2]) * plotx)

plt.show()