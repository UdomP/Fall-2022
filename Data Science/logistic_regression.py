# Code source: Gael Varoquaux
# License: BSD 3 clause
# Modified from: https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html#sphx-glr-auto-examples-linear-model-plot-logistic-py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from scipy.special import expit

# Generate a toy dataset, it's just a straight line with some Gaussian noise:
n_samples = 100
np.random.seed(0)
X = 100*np.random.normal(size=n_samples)
y = (X > 70).astype(float)
X += 3 * np.random.normal(size=n_samples) # random noise

X = X[:, np.newaxis]

# Fit the Logistic Regression: y = 1/[1+e^(-(beta0 + beta1*x))]
clf = LogisticRegression()
clf.fit(X, y)

# plot the result: numpy.ravel() returns a contiguous flattened array.
plt.scatter(X.ravel(), y, color="black")
X_test = np.linspace(-100, 100, 5000)

# Expit(): logistic sigmoid function: f(z) = 1/(1+e^(-z))
print("Coefficient, beta1: \n", clf.coef_)
print("Intercept, beta0: \n", clf.intercept_)
logistic_sigmoid = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, logistic_sigmoid, color="red", linewidth=3)
plt.show()