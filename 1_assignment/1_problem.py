# SGD In Python!
"""

General flow:
1. Weights and biases of the model are the starting values that the algo starts with,
    in order to improve the acc. the model's preds, these params are changed during training.
2. From the training dataset, a random or small number of data points (thus "stochastic") is chosen at the beginning of each cycle
3. Algo determines the grad of the loss func w/r/t model params for the chosen data point or dps. The diff in error between goal
    values and the anticipated values is measured by the loss function
4. Then the model updates params

Algo behind SGD
1. Init: The model params with some initial values, likely small
2. Using loss function. algo implements some loss func to minimize the loss which quantifies the difference between the models' preds and actual target values
3. Stochasticity: The algo uses one data example at a time instead of the whole dataset
4. Gradient calc: Calculate gradient of the loss with respect to model params
5. Updating Parameters: after computing the gradient, update the params of the algo
6. Iterate: Steps 4-5 are done for multiple data points and each iteration is called an epoch
7. Converge: This continues until a stopping criteria is met like max/min

"""

# Beginning with ScKit-Learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

california = fetch_california_housing()
X = california.data
Y = california.target

# Split the data into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=63)

# Implementing SGD Regressor
sgd_regressor = SGDRegressor(max_iter=1000, alpha=0.0001, learning_rate='invscaling', random_state=63)

# Fit the training data and predict using the test data
sgd_regressor.fit(X_train, y_train)
y_pred = sgd_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"SK_learn: MSE is: {mse}")

# Hand-rolled Gradient Descent
def gradient_descent(X, y, lr=0.01, epoch=1000):
    m, b = 0.2, 0.2     # Parameters
    log, mse = [], []   # Lists to store the learning process
    N = len(X)          # Number of samples
    for _ in range(epoch):
        f = y - (m*X + b)
        
        # Updating m and b
        m -= lr * (-2 * X.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        log.append((m,b))
        mse.append(mean_squared_error(y, (m*X + b)))
    return m, b, log, mse


housing_data = california
Features = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
Target = pd.DataFrame(housing_data.target, columns=['Target'])
df = Features.join(Target)
df.corr()
df[['MedInc', 'Target']].describe()[1:] #.style.highlight_max(axis=0)
df = df[df.Target < 3.5]
df = df[df.MedInc < 8]
df[['MedInc', 'Target']].describe()[1:]
def scale(x):
    min = x.min()
    max = x.max()
    return pd.Series([(i - min)/(max - min) for i in x])

X = scale(df.MedInc)
y = scale(df.Target)

X = df.MedInc
y = df.Target
m, b, log, mse = gradient_descent(X, y, lr=0.01, epoch=100)

y_pred = m*X + b

print(f"Manual MSE: {mean_squared_error(y,y_pred)}")

# Running the SGD Iteration
def run_sgd(X, y, lr=0.01, epoch=1000, batch_size=1):
    m, b = 0.5, 0.5
    log, mse = [], []
    for _ in range(epoch):
        indexes = np.random.randint(0, len(X), batch_size) # random sample
        #print(f"Indexes: {indexes}")
        Xs = np.take(a=X, indices=indexes, axis=0)
        ys = np.take(a=y, indices=indexes, axis=0)
        N = len(Xs)
        
        f = ys - (m*Xs + b)
        
        m -= lr * (-2 * Xs.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        log.append((m,b))
        mse.append(mean_squared_error(y, (m*X + b)))
    return m, b, log, mse

m, b, log, mse = run_sgd(X, y, lr=0.01, epoch=100, batch_size=2)

y_pred = m*X + b
print(f"Manual SGD MSE: {mean_squared_error(y,y_pred)}")
