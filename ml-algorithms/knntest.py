import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print(X_train.shape)
print(X_test.shape)

print(X_train[0])

print(y_train.shape)
print(y_train)

plt.figure()
plt.scatter(X[ :, 0], X[ :, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

a = [1, 2, 1, 4, 5, 4, 3, 2, 1, 2, 3, 3, 4, 5, 4, 3, 2, 1, 2, 3]
from collections import Counter
most_common = Counter(a).most_common(1)
print(most_common[0])

from knn import KNN
vlf = KNN(k=3)
vlf.fit(X_train, y_train)
predictions = vlf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print("KNN classification accuracy: ", acc)