import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import os

from SimplePerceptron import SimplePerceptron
from Plot import data_plot, plot_errors_vs_epochs, plot_decision_boundary
from Utils import generate_linearly_separable_data, generate_not_so_linearly_separable_data

# Create the output folder if it doesn't exist
if not os.path.exists("Output/ex1"):
    os.makedirs("Output/ex1/")


X, y = generate_not_so_linearly_separable_data(50)
y = np.where(y == 0, -1, 1)

# Plot the data
data_plot(X, y, "Linearly not separable data", "X", "Y", "Output/ex1/not_linearly_separable_data.png")

# Train the perceptron
perceptron = SimplePerceptron(learning_rate=0.01, epochs=1000, accepctable_error=0.01)
epochs, error, errors = perceptron.fit(X, y)

print("Epochs: ", epochs)
print("Errors", errors)

# Plot the error over the epochs
plot_errors_vs_epochs(epochs, errors, "Output/ex1/error_over_the_epochs_not_separable.png")

# Plot decision boundary
plot_decision_boundary(X, y, perceptron.weights[0], perceptron.weights[1], perceptron.bias, "Output/ex1/perceptron_decision_boundary_not_separable.png", "Perceptron Decision Boundary")

# Use the custom support vector machine
from SVM import SVM

X, y = generate_linearly_separable_data(50)
y = np.where(y == 0, -1, 1)

svm = SVM(learning_rate=0.001, c=1, epochs=1000)
svm.fit(X, y)

margin = svm.margin() / 2

def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]

def plot_svm(margin, X, y, svm, filename):
    
    ax = plt.subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, svm.weights, svm.bias, 0)
    x1_2 = get_hyperplane_value(x0_2, svm.weights, svm.bias, 0)


    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k', label="C=" + str(svm.c))

    ax.plot([x0_1, x0_2], [x1_1 - margin, x1_2 - margin], 'b--')
    ax.plot([x0_1, x0_2], [x1_1 + margin, x1_2 + margin], 'b--')

    ymin = np.amin(X[:, 1])
    ymax = np.amax(X[:, 1])
    ax.set_ylim([ymin-3, ymax+3])

    plt.title("SVM Decision Boundary")
    plt.legend()
    plt.savefig(filename)
    plt.show()

# Plot decision boundary
plot_svm(margin, X, y, svm, "Output/ex1/decision_boundary_svm_separable.png")


X, y = generate_not_so_linearly_separable_data(50)
y = np.where(y == 0, -1, 1)

svm = SVM(learning_rate=0.001, c=1, epochs=1000)
svm.fit(X, y)

margin = svm.margin() / 2
# Plot decision boundary
plot_svm(margin, X, y, svm, "Output/ex1/decision_boundary_svm_not_separable.png")

# try with different values of C

svm = SVM(learning_rate=0.001, c=0.1, epochs=1000)
svm.fit(X, y)

margin = svm.margin() / 2
# Plot decision boundary
plot_svm(margin, X, y, svm, "Output/ex1/decision_boundary_svm_not_separable_c_0_1.png")

svm = SVM(learning_rate=0.001, c=0.01, epochs=1000)
svm.fit(X, y)

margin = svm.margin() / 2

# Plot decision boundary
plot_svm(margin, X, y, svm, "Output/ex1/decision_boundary_svm_not_separable_c_0_01.png")

