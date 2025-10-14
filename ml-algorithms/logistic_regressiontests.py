import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(X_train.shape, X_test.shape)

regressor = LogisticRegression(n_lr=0.0001, n_iter=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("LR classification accuracy:", accuracy_score(y_test, predictions))

# Plotting some data points
plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel(bc.feature_names[0])
plt.ylabel(bc.feature_names[1])
plt.title("Breast Cancer Data - First Two Features")
plt.show()
