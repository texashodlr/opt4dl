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
print(f"MSE is: {mse}")


