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

### 1 Dimensional ###

def synthetic_simple_lr_data(n=100, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    w_true, b_true = 2.0, -1.0              # Correct/true data
    X = rng.uniform(-5, 5, size=n)          # X is the features vector
    y = w_true * X + b_true + noise * rng.normal(size=n)    # Noise Line
    print("### Synthetic Data Function Call ###")
    print(f"Feature Vectors X: {X}, Len(X): {len(X)}")
    print(f"Label Vector y: {y}, Len(y): {len(y)}")
    return X, y, w_true, b_true

def simple_linear_regression(X, y, lr=0.01, epochs=50):
    # 1-D Linear regression (X)
    n = len(X)
    w, b = 0.0, 0.0
    for epoch in range(epochs):
        for i in range(n):
            x_i, y_i = X[i], y[i]
            y_hat = w * x_i + b
            error = y_hat - y_i
            dw = error * x_i        # Gradient 'w'
            db = error              # Gradient 'b'
            w -= lr * dw            # Update 'w'
            b -= lr * db            # Update 'b'
        if epoch % 10 == 0:
            mse = np.mean((w * X + b - y) ** 2)
            print(f"Epoch {epoch}: w={w:.3f}, b={b:.3f}, MSE={mse:.3f}")
    return w, b
### 1 Dimensional ###
### N-Dimensional ###
def synthetic_nd_lr_data(n=100, d=3, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d)             # 
    b_true = rng.normal()                   # Correct/true data (scalar)
    print(f"Printing True w and b values")
    print(f"True w: {w_true}, Len(w): {len(w_true)}")
    print(f"True b: {b_true}")
    X = rng.normal(size=(n,d))          # X is the features vector
    y = X @ w_true + b_true + noise * rng.normal(size=n)    # Noise Line
    print("### N-d Synthetic Data Function Call ###")
    print(f"Feature Vectors X: {X}, Len(X): {len(X)}, Len(X[0]): {len(X[0])}")
    print(f"Label Vector y: {y}, Len(y): {len(y)}")
    return X, y, w_true, b_true

def simple_sgd_linear_regression(X, y, lr=1e-2, epochs=20, batch_size=128, l2=0.0, seed=0, standardize=True):
    """
    X: feature matrix [n, d] {n samples and d features}
    y: label values (vector of length n)
    lr (eta): learning rate
    epochs: how many times to iterate through the dataset
    batch_size: how many samples per sgd update (minibatch)
    l2: L2 regularization strength, ridge penalty
    seed: RNG seed for reproducibility
    mu: mean of each feature
    sigma: std of each feature
    
    """
    rng = np.random.default_rng(seed)
    X = X.astype(float, copy=True)
    y = y.astype(float, copy=True)
    # Standardizing
    mu = np.zeros(X.shape[1])
    sigma = np.ones(X.shape[1])
    
    n, d = X.shape
    w = np.zeros(d)                     # W init'd as zeros (d weights so 12 ex)
    b = 0.0                             # b initd as a scalar zero
    
    # Training loop where we shuffle 
    for _ in range(epochs):
        idx = rng.permutation(n)        # Shuffles the set
        Xs, ys = X[idx], y[idx]
        for start in range(0, n, batch_size): # chunking the data into batches of xb(atch) and yb(atch)
            xb = Xs[start:start+batch_size]
            yb = ys[start:start+batch_size]
            m = xb.shape[0]
            
            pred = xb @ w + b           # Prediction scalar y_hat = Xb*w + b
            err = pred - yb             # Error vector -> err = y_hat - y
            
            dw = (xb.T @ err) / m   # Gradients with respect to weights
                                    # Grad_w = 1/m * X_b.T (X_b*w + b - y_b)
            db = err.mean()         # Gradient ""   ""          "" Bias (scalar)
                                    # Grad_b = 1/m * [SUM(i->m) (W.t*x_i + b - y_i)]
            
            if l2 > 0:
                dw += l2 * w
            
            w -= lr * dw
            b -= lr * db
    w_orig = w / sigma
    b_orig = b - (mu @ w_orig)
    return w_orig, b_orig


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
def call_1():
    X, y, w_true, b_true = make_fake_data()
    print("True weights:", w_true, "bias:", b_true)
    print("Shapes -> X:", X.shape, " y:", y.shape)
    w = sgd_linear_regression(X, y)
    print("SGD Weights: ", w)

# Call 2
def call_2():
    # 1-Dimensional LR code
    X, y, w_true, b_true = synthetic_simple_lr_data(n=100000, noise=0.5, seed=63)
    w_hat, b_hat = simple_linear_regression(X, y, lr=0.001, epochs=100)
    print("True parameters:  w =", w_true, " b =", b_true)
    print("Learned parameters: w =", w_hat, " b =", b_hat)

def call_3():
    X, y, w_true, b_true = synthetic_nd_lr_data(n=100000, d=12, noise=0.5, seed=63)
    w_hat, b_hat = simple_sgd_linear_regression(X, y, lr=5e-2, epochs=30, batch_size=256, l2=1e-3, seed=36)

    print("True  bias:", round(b_true, 3), " | Learned bias:", round(b_hat, 3))
    print("True  w[:5]:", np.round(w_true[:5], 3))
    print("Learn w[:5]:", np.round(w_hat[:5], 3))

    mse = np.mean((X @ w_hat + b_hat - y)**2)
    print("Train MSE:", round(mse, 4))
    
call_3()
