from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from builtins import enumerate

def load_data():
    train = np.genfromtxt('quasar_train.csv', skip_header=True, delimiter=',')
    test = np.genfromtxt('quasar_test.csv', skip_header=True, delimiter=',')
    wavelengths = np.genfromtxt('quasar_train.csv', skip_header=False, delimiter=',')[0]
    return train, test, wavelengths

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
  
    X[:, 0].fill(1)
    X[:, 1:] = X_
    return X

def smooth_data(raw, wavelengths, tau):
    return LWR_smooth(raw.T, wavelengths.reshape((wavelengths.size, 1)), tau)

def LWR_smooth(spectrum, wavelengths, tau):
    smooth_spectrum = np.zeros(spectrum.shape)
    X = add_intercept(wavelengths)
    for c, i in enumerate(wavelengths):
        W = np.diag(np.exp(-(wavelengths.reshape((wavelengths.shape[0],)) - i) ** 2 / (2 * tau ** 2)))
        theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ spectrum
        smooth_spectrum[c, :] = X[c, :] @ theta
    return smooth_spectrum

def LR_smooth(Y, X_):
    X = add_intercept(X_)
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y
    yhat = X @ theta
    return yhat, theta

def plot_b(X, raw_Y, Ys, desc, filename):
    for c, i in enumerate(Ys):
        plt.figure()
        plt.title(desc[c])
        plt.scatter(X, raw_Y)
        plt.plot(X, i, color='red', linewidth=2)
        plt.show()
        plt.savefig(filename)
    return

def plot_c(Yhat, Y, X, filename):
    plt.figure()
    plt.plot(X, Y, color='blue', linewidth=2)
    plt.plot(X[0:Yhat.size], Yhat, color='red', linewidth=2)
    plt.show()
    plt.savefig(filename)
    return

def split(full, wavelengths):
    # wavelengths is 1D, full is numWavelengths x numObservations
    leftNum = wavelengths[wavelengths < 1200].size
    rightNum = wavelengths[wavelengths >= 1300].size
    # returns left and right subarrays
    return full[-leftNum : , :], full[ : rightNum, :]

def dist(a, b):
    return np.sum(np.square(a - b), axis=0)

def func_reg(left_train, right_train, right_test):
    k = 3
    right_test = right_test.reshape((right_test.size, 1))
    
    distances_to_current = dist(right_train, right_test)
    closest_indices = np.argsort(distances_to_current)[:k]
    h = np.amax(distances_to_current)
    closest_kers = 1 - (np.take(distances_to_current, closest_indices) / h)
    closest_f_left = np.take(left_train, closest_indices, axis=1)
    # returns f_left_hat
    return np.sum(closest_f_left * closest_kers, axis=1) / np.sum(closest_kers)

def main():
    raw_train, raw_test, wavelengths = load_data()
    ## Part b.i
    lr_est, theta = LR_smooth(raw_train[0].reshape((raw_train[0].size, 1)),
                              wavelengths.reshape((wavelengths.size, 1)))
    print('Part b.i) Theta=[%.4f, %.4f]' % (theta[0], theta[1]))
    plot_b(wavelengths.reshape((wavelengths.size, 1)),
           raw_train[0].reshape((raw_train[0].size, 1)),
           [lr_est], ['Regression line'], 'ps1q5b1.png')

    ## Part b.ii
    lwr_est_5 = LWR_smooth(raw_train[0].reshape((raw_train[0].size, 1)),
                           wavelengths.reshape((wavelengths.size, 1)), 5)
    plot_b(wavelengths.reshape((wavelengths.size, 1)),
           raw_train[0].reshape((raw_train[0].size, 1)),
           [lwr_est_5], ['tau = 5'], 'ps1q5b2.png')

    ## Part b.iii
    lwr_est_1 = LWR_smooth(raw_train[0].reshape((raw_train[0].size, 1)),
                           wavelengths.reshape((wavelengths.size, 1)), 1)
    lwr_est_10 = LWR_smooth(raw_train[0].reshape((raw_train[0].size, 1)),
                            wavelengths.reshape((wavelengths.size, 1)), 10)
    lwr_est_100 = LWR_smooth(raw_train[0].reshape((raw_train[0].size, 1)),
                             wavelengths.reshape((wavelengths.size, 1)), 100)
    lwr_est_1000 = LWR_smooth(raw_train[0].reshape((raw_train[0].size, 1)),
                              wavelengths.reshape((wavelengths.size, 1)), 1000)
    plot_b(wavelengths.reshape((wavelengths.size, 1)),
           raw_train[0].reshape((raw_train[0].size, 1)),
             [lwr_est_1, lwr_est_5, lwr_est_10, lwr_est_100, lwr_est_1000],
             ['tau = 1', 'tau = 5', 'tau = 10', 'tau = 100', 'tau = 1000'],
             'ps1q5b3.png')

    ## Part c.i
    smooth_train, smooth_test = [smooth_data(raw, wavelengths, 5) for raw in [raw_train, raw_test]]

    ## Part c.ii
    left_train, right_train = split(smooth_train, wavelengths)
    left_test, right_test = split(smooth_test, wavelengths)

    train_errors = [dist(left, func_reg(left_train, right_train, right)) for left, right in zip(left_train.T, right_train.T)]
    print('Part c.ii) Training error: %.4f' % np.mean(train_errors))

    ## Part c.iii
    test_errors = [dist(left, func_reg(left_train, right_train, right)) for left, right in zip(left_test.T, right_test.T)]
    print('Part c.iii) Test error: %.4f' % np.mean(test_errors))

    left_1 = func_reg(left_train, right_train, right_test[:,0])
    plot_c(left_1, smooth_test[:,0], wavelengths, 'ps1q5c3_1.png')
    left_6 = func_reg(left_train, right_train, right_test[:,5])
    plot_c(left_6, smooth_test[:,5], wavelengths, 'ps1q5c3_6.png')


if __name__ == '__main__':
    main()
