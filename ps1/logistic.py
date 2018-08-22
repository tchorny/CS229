import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

alpha = 0.001
theta = np.zeros(3)

def sigmoid(theta, x):
    return 1.0 / (1.0 + math.exp(-np.dot(theta, x)))


xdata = pd.read_csv("logistic_x.txt", sep='\s+', header=None)
ydata = pd.read_csv("logistic_y.txt", sep='\s+', header=None)

x = xdata.values
y = ydata.values
intercept = np.ones((y.size, 1))
x = np.concatenate((intercept, x), axis=1)

for i in range(len(y)):
    theta += alpha * (y[i,:] - sigmoid(theta, x[i,:])) * x[i,:]
    
print(theta)
#plt.scatter(xdata.values[:, 0], xdata.values[:, 1], c=ydata.values[:, 0])
#plt.show()