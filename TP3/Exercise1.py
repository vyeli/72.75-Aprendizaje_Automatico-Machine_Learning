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
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=2, cluster_std=1.5, random_state=42)
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
# w1x1 + w2x2 + b = 0 -> x2 = (-w1x1 - b) / w2 -> y = mx + b
# Ax + By + C = 0 -> y = (-Ax - C) / B

def plot_decision_boundary(X, y, w1, w2, b):
    ax = plt.subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = (-w1 * x0_1 - b) / w2
    x1_2 = (-w1 * x0_2 - b) / w2

    ymin = np.amin(X[:, 1])
    ymax = np.amax(X[:, 1])
    ax.set_ylim([ymin-3, ymax+3])

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k', label="Perceptron Decision Boundary")
    plt.legend()
    plt.savefig("Output/ex1/decision_boundary.png")
    plt.show()

plot_decision_boundary(X, y, perceptron.weights[0], perceptron.weights[1], perceptron.bias)

# Obtain the hyperplane that maximaze the margin base on the decision boundary of the perceptron
# Choose possibles support vectors

# Calculate the distance of each point from the decision boundary
# distance = |Ax + By + C| / sqrt(A^2 + B^2)
distances = np.abs(np.dot(X, perceptron.weights) + perceptron.bias) / np.sqrt(np.dot(perceptron.weights, perceptron.weights))

sorted_distances = np.argsort(distances)
# Select 3 points, 2 of them will be the support vectors and the other one will be the point that maximizes the margin
support_vectors = X[sorted_distances[:5]]

# For each support vector calculate the margin and select the one that maximizes it
# margin = 2 / sqrt(A^2 + B^2)

max_margin = 0
best_line = None
best_point = None

# Iterate over each pair of points
for i in range(len(support_vectors)):
    for j in range(i + 1, len(support_vectors)):
        # Check if the points are from the same class
        if y[sorted_distances[i]] != y[sorted_distances[j]]:
            continue

        # Calculate the line between the two points
        point1 = support_vectors[i]
        point2 = support_vectors[j]
        line = np.polyfit([point1[0], point2[0]], [point1[1], point2[1]], 1)
        # line = [A, B] -> y = Ax + B
        
        # Iterate over the remaining points to find the one that maximizes the margin
        for k in range(len(support_vectors)):
            if k != i and k != j and y[sorted_distances[k]] != y[sorted_distances[i]]:
                point = support_vectors[k]
                # Calculate the distance from the point to the line
                # distance = |Ax - y + B| / sqrt(A^2 + 1)
                distance = np.abs(line[0] * point[0] - point[1] + line[1]) / np.sqrt(line[0]**2 + 1)
                # If the distance is greater than the current maximum margin, update the maximum margin and the best line and point
                if distance > max_margin:
                    max_margin = distance
                    best_line = line
                    best_point = point

# Plot the best line with its margin
margin = max_margin / 2

ax = plt.subplot(1, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=y)

x0_1 = np.amin(X[:, 0])
x0_2 = np.amax(X[:, 0])

x1_1 = best_line[0] * x0_1 + best_line[1]
x1_2 = best_line[0] * x0_2 + best_line[1]

ymin = np.amin(X[:, 1])
ymax = np.amax(X[:, 1])
ax.set_ylim([ymin-3, ymax+3])

ax.plot([x0_1, x0_2], [x1_1 - margin, x1_2 - margin], 'b', label="Optimal line with margin")

# Plot the margin lines

ax.plot([x0_1, x0_2], [x1_1, x1_2], 'b--')

ax.plot([x0_1, x0_2], [x1_1 - max_margin, x1_2 - max_margin], 'b--')

# plot old decision boundary

x1_1 = (-perceptron.weights[0] * x0_1 - perceptron.bias) / perceptron.weights[1]
x1_2 = (-perceptron.weights[0] * x0_2 - perceptron.bias) / perceptron.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], 'r', label="Perceptron")

plt.title("Best line with its margin")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig("Output/ex1/best_line.png")
plt.show()

