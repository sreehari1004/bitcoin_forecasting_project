from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

class RBFNN:
    def __init__(self, n_centers=10):
        self.n_centers = n_centers
        self.centers = None
        self.spread = None
        self.linear_model = None

    def _rbf(self, X):
        # Gaussian RBF
        return np.exp(-np.linalg.norm(X[:, None] - self.centers, axis=2) ** 2 / (2 * self.spread ** 2))

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.n_centers, random_state=42).fit(X)
        self.centers = kmeans.cluster_centers_
        d_max = np.max([np.linalg.norm(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.spread = d_max / np.sqrt(2 * self.n_centers)

        Phi = self._rbf(X)
        self.linear_model = LinearRegression().fit(Phi, y)

    def predict(self, X):
        Phi = self._rbf(X)
        return self.linear_model.predict(Phi)

def train_rbfnn(X_train, y_train, n_centers=10):
    model = RBFNN(n_centers=n_centers)
    model.fit(X_train, y_train)
    return model