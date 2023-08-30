
import numpy as np
from sklearn.datasets import make_regression
from matplotlib import pyplot as mpl

## Dataset:
n_s = 100 # number of samples is by default 100
n_f = 1
noise = 7
# n_t = 1 # is 1 by default
F, T = make_regression(n_features=n_f, noise=noise)
    # F for features matrix, T for Target 

# mpl.scatter(F, T)
# mpl.show()

# print(F.shape) -> (100, 1)
# print(T.shape) -> (100,)
T = T.reshape(T.shape[0], 1)

## Model:
n_mc = 2 # number of model coefficients
M = np.hstack((F, np.ones(F.shape)))
    # remember hstack() takes 1 argument for that we add parentheses.
# print(M)
X = np.random.randn(n_mc, 1)
# print(X)
def model(M, X):
    return M.dot(X)

# print(model(M, X))
mpl.scatter(F, T)
mpl.plot(F, model(M, X), c = 'r')
mpl.show()

## Cost Function:
def cost_function(M, T, X):
    return 1/(2*n_s)*np.sum((T - model(M, X))**2)

# print(cost_function(M, T, X))

## Gradient Descent Algorithm:
def gradient(M, T, X):
    return 1/n_s * M.T.dot(model(M, X) - T)

def gradient_descent(M, T, X, learning_rate, n_iterations):
    cost = np.zeros(n_iterations) 
    for i in range(0, n_iterations):
        X = X - learning_rate * gradient(M, T, X) 
        cost[i] = cost_function(M, T, X)
        
    return X, cost

n_iterations = 10**(5)
learning_rate = 10**(-2)

X, cost = gradient_descent(M, T, X, learning_rate, n_iterations)


pred = model(M, X)

mpl.scatter(F, T)
mpl.plot(F, pred, c='r')
mpl.show()
