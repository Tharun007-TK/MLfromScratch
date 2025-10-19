# perceptron.py
import numpy as np


class Perceptron:
    """
    A simple implementation of the Perceptron algorithm.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000, verbose=False):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.verbose = verbose
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the Perceptron on the dataset.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Ensure labels are 0 or 1
        y_ = np.where(y <= 0, 0, 1)

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

            if self.verbose and i % (self.n_iters // 10) == 0:
                print(f"Iteration {i}/{self.n_iters} | Weights: {self.weights} | Bias: {self.bias}")

    def predict(self, X):
        """
        Predict class labels for input data.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        """
        Activation function: step function.
        """
        return np.where(x >= 0, 1, 0)


# -----------------------------
# TESTING SECTION
# -----------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    # Generate a simple 2D dataset
    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Initialize and train the perceptron
    p = Perceptron(learning_rate=0.01, n_iters=1000, verbose=True)
    p.fit(X_train, y_train)

    # Make predictions
    predictions = p.predict(X_test)
    acc = accuracy(y_test, predictions)
    print(f"\n✅ Perceptron Classification Accuracy: {acc * 100:.2f}%")

    # Plot decision boundary
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train, cmap="coolwarm")

    x0_min, x0_max = np.amin(X_train[:, 0]), np.amax(X_train[:, 0])
    x1_min = (-p.weights[0] * x0_min - p.bias) / p.weights[1]
    x1_max = (-p.weights[0] * x0_max - p.bias) / p.weights[1]

    ax.plot([x0_min, x0_max], [x1_min, x1_max], "k", label="Decision Boundary")
    ax.legend()
    ax.set_title("Perceptron Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    plt.show()
