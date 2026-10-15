import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.asarray(X)])

    def _predict_one(self, x):
        # Euclidean distance from x to every training point
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # indices of the k closest points
        nearest = np.argsort(distances)[:self.k]
        # majority vote among their labels
        return Counter(self.y_train[nearest]).most_common(1)[0][0]