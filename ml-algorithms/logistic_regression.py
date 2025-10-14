import numpy as np

class LogisticRegression:

    def __init__(self, n_lr = 0.001, n_iter = 1000):
        self.lr = n_lr
        self.iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            dp = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * dp

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
if __name__ == "__main__":
    # Example usage
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    print(X_train.shape, X_test.shape)

    regressor = LogisticRegression(n_lr=0.0001, n_iter=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    print("LR classification accuracy:", accuracy_score(y_test, predictions))
     