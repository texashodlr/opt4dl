# Linear Regression with SGD in Python
"""
Main idea behind Linear Regression: fit a line that is the best fit for the data.
    Do this via the Least Square Method
https://www.turing.com/kb/beginners-guide-to-complete-mathematical-intuition-behind-linear-regression-algorithms
"""

## Linear Regression Model
"""
 Features x epsilon R^d
 Target   y epsilon R
 
 y_hat = w_T*x+b
 
 Objective Loss: Using MSE with L2 regularization
    L(w~) = 1/2n sigma (i=1 -> n)[w_~T * x_~_sub_i - y_sub_i)^2 + Lambda/2*norm[[w]]^2

 Gradients
    grad_w~L = 1/n X_~T(X_~*w_~-y) + Lambda*r_~

SGD Update:
    w_~ = w_~ - eta*g

"""

import numpy as np

def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

def sgd_linear_regression(X, y, lr = 1e-2, epochs=20, batch_size=32, l2=0.0, seed=0):
    rng  = np.random.default_rng(seed)
    X = add_bias(X)     # [n, d+1] w/ last col = 1 for bias
    n, d1 = X.shape
    w = np.zeros(d1)    # Includes bias
    for _ in range(epochs):
        idx = rng.permutation(n)
        X, y = X[idx], y[idx]
        for start in range(0, n, batch_size):
            xb = X[start:start+batch_size]
            yb = y[start:start+batch_size]
            # Predictions and residual data
            pred = xb @ w
            err = pred - yb
            # MSE Gradient
            g = (xb.T @ err) / xb.shape[0]
            # L2 on weights w/out bias
            if l2 > 0:
                g[:-1] += l2 * w[:-1]
            # Updating
            w -= lr * g
    return w

def closed_form_ridge(X, y, l2=0.0):
    Xb = add_bias(X)
    d1 = Xb.shape[1]
    I = np.eye(d1); I[-1, -1] = 0.0
    return np.linalg.solve(Xb.T @ Xb + l2 * I, Xb.T @ y)

def make_fake_data(n=1000, d=3, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    # True weights (d features, and bias)
    w_true = rng.normal(size=d)
    b_true = rng.normal()
    # Features
    X = rng.normal(size=(n,d))
    y = X @ w_true + b_true + noise * rng.normal(size=n)
    return X, y, w_true, b_true

# Initial Iteration
X, y, w_true, b_true = make_fake_data()
print("True weights:", w_true, "bias:", b_true)
print("Shapes -> X:", X.shape, " y:", y.shape)
w = sgd_linear_regression(X, y)
print("SGD Weights: ", w)