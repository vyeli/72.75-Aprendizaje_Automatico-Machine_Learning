import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import os

from SimplePerceptron import SimplePerceptron

# Create the output folder if it doesn't exist
if not os.path.exists("Output/ex1"):
    os.makedirs("Output/ex1/")

# Generates linearly separable data 
# REMEMBER THE RANDOMSTATE USED TO RECREATE THE IMAGE
def generate_linearly_separable_data(n_samples):
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=2, cluster_std=1.05, random_state=42)
    return X, y

def generate_not_so_linearly_separable_data(n_samples):
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=2, cluster_std=3, random_state=42)
    return X, y

X, y = generate_linearly_separable_data(50)
y = np.where(y == 0, -1, 1)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Linearly separable data")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("Output/ex1/linearly_separable_data.png")
plt.show()



# Train the perceptron
perceptron = SimplePerceptron(learning_rate=0.01, epochs=1000, accepctable_error=0.01)
epochs, error, errors = perceptron.fit(X, y)

print("Epochs: ", epochs)
print("Errors", errors)


# Plot the error over the epochs
plt.plot(range(epochs), errors)
plt.title("Error over the epochs")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.savefig("Output/ex1/error_over_the_epochs.png")
plt.show()


# Plot decision boundary
# w1x1 + w2x2 + b = 0 -> x2 = (-w1x1 - b) / w2

ax = plt.subplot(1, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=y)

x0_1 = np.amin(X[:, 0])
x0_2 = np.amax(X[:, 0])

x1_1 = (-perceptron.weights[0] * x0_1 - perceptron.bias) / perceptron.weights[1]
x1_2 = (-perceptron.weights[0] * x0_2 - perceptron.bias) / perceptron.weights[1]

ymin = np.amin(X[:, 1])
ymax = np.amax(X[:, 1])
ax.set_ylim([ymin-3, ymax+3])

ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

plt.savefig("Output/ex1/decision_boundary.png")
plt.show()

# Obtain the hyperplane that maximaze the margin base on the decision boundary of the perceptron
# Choose possibles support vectors

distances = np.abs((np.dot(X, perceptron.weights) + perceptron.bias) / np.linalg.norm(perceptron.weights))
support_vector_indices = np.where(distances == min(distances))
support_vectors = X[support_vector_indices]

