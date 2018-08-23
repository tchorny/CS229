from __future__ import division
import numpy as np
import matplotlib as mpl
from matplotlib.pyplot import axis
#mpl.use('Agg')
import matplotlib.pyplot as plt

def load_data():
    X = np.genfromtxt('logistic_x.txt')
    Y = np.genfromtxt('logistic_y.txt')
    return X, Y

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
  
    X[:, 0].fill(1)
    X[:, 1:] = X_

    return X

def calc_grad(X, Y, theta):
    m, n = X.shape

    grad = (np.sum((-1.0 / m) * (1 / (1 + np.exp(Y * (X @ theta))))
                   * Y * X, axis=0)).reshape((n, 1))

    return grad

def calc_loss(X, Y, theta):
    m, n = X.shape

    return np.sum((1/m) * np.log(1 + np.exp(-Y * (X @ theta))))

def calc_hessian(X, Y, theta):
    m, n = X.shape
    H = np.zeros((n, n))

    temp = np.exp(Y * (X @ theta))
    for i in range(m):
        Xi = X[i, :].reshape((1, n))
        H += (1.0 / m) * (Xi.T @ Xi) * (temp[i] / (1 + temp[i])**2)

    return H

def logistic_regression(X, Y):
    epsilon = 0.0001
    m, n = X.shape
    theta = np.zeros((n, 1))
    delta_loss = 1.0

    while(abs(delta_loss) > epsilon):
        loss_0 = calc_loss(X, Y, theta)
        grad = calc_grad(X, Y, theta)
        inv_hessian = np.linalg.inv(calc_hessian(X, Y, theta))
        theta -= inv_hessian @ grad
        delta_loss = calc_loss(X, Y, theta) - loss_0

    return theta

def plot(X, Y, theta):
    plt.figure()

    plt.scatter(X[:, 1], X[:, 2], c=Y[:, 0])
    plotx = np.arange(np.amin(X[:, 1]), np.amax(X[:, 1]), 0.01)
    plt.plot(plotx, -theta[0, 0]/theta[2, 0] - (theta[1, 0]/theta[2, 0]) * plotx)

    plt.show()

    plt.savefig('ps1q1c.png')
    return

def main():
    X_, Y = load_data()
    Y = Y.reshape((Y.size, 1))
    X = add_intercept(X_)
    theta = logistic_regression(X, Y)
    print(theta)
    plot(X, Y, theta)

if __name__ == '__main__':
    main()
