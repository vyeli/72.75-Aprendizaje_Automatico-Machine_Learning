from sklearn.datasets import make_blobs
import numpy as np
from KMeans import KMeans
from sklearn.datasets import load_iris
from Kohonen import Kohonen

np.random.seed(42)

X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)

print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(K=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Normalize the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Train the Kohonen network
kohonen = Kohonen(map_size=(15, 15), n_features=X.shape[1], n_iterations=5000, random_seed=40)
kohonen.fit(X)

kohonen.plot_u_matrix()