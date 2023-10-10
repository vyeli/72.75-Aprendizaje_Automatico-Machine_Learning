import numpy as np
import matplotlib.pyplot as plt


def data_plot(X, y, title, xlabel, ylabel, filename):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.show()

def plot_errors_vs_epochs(epochs, errors, filename="Output/ex1/error_over_the_epochs.png"):
    plt.plot(range(epochs), errors)
    plt.title("Error over the epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.savefig(filename)
    plt.show()

# Plot the decision boundary
# w1x1 + w2x2 + b = 0
# x2 = (-w1x1 - b) / w2 => y = (-w1x - b) / w2
# Ax + By + C = 0
def plot_decision_boundary(X, y, w1, w2, b, filename="Output/ex1/decision_boundary.png", title="Decision boundary"):
    ax = plt.subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = (-w1 * x0_1 - b) / w2
    x1_2 = (-w1 * x0_2 - b) / w2

    ymin = np.amin(X[:, 1])
    ymax = np.amax(X[:, 1])
    ax.set_ylim([ymin-3, ymax+3])

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
    plt.title(title)
    plt.savefig(filename)
    plt.show()
