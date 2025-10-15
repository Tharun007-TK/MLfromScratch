import numpy as np
from naivebayes import NaiveBayes
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import unittest


class TestNaiveBayes(unittest.TestCase):
	def test_iris_dataset(self):
		# Load iris dataset
		iris = load_iris()
		X, y = iris.data, iris.target
		
		# Split the data
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42)
		
		# Train the model
		model = NaiveBayes()
		model.fit(X_train, y_train)
		
		# Make predictions
		predictions = model.predict(X_test)
		
		# Check accuracy
		accuracy = accuracy_score(y_test, predictions)
		self.assertGreater(accuracy, 0.8, "Accuracy should be above 80%")
		
	def test_simple_dataset(self):
		# Create a simple dataset
		X = np.array([[1, 1], [1, 2], [2, 1], [5, 5], [5, 6], [6, 5]])
		y = np.array([0, 0, 0, 1, 1, 1])
		
		# Train the model
		model = NaiveBayes()
		model.fit(X, y)
		
		# Test predictions
		self.assertEqual(model.predict([[1, 1]]), 0)
		self.assertEqual(model.predict([[5, 5]]), 1)


if __name__ == "__main__":
	unittest.main()

