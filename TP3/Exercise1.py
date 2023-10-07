import numpy as np
import matplotlib.pyplot as plt

# Generates linearly separable data
def generate_linearly_separable_data(n_samples):
    np.random.seed(0)
    X = np.random.uniform(0, 5, (n_samples, 2))
    y = np.where(X[:, 0] > X[:, 1], 1, -1)
    return X, y

X, y = generate_linearly_separable_data(100)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Linearly separable data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

