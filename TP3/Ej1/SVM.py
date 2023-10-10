import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, c=0.5, epochs=1000):
        self.lr = learning_rate
        self.c = c
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (1 / (2 * self.c) * self.weights)
                else:
                    self.weights -= self.lr * (1 / (2 * self.c) * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.lr * y[idx]


    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
    
    def margin(self):
        return 2 / np.linalg.norm(self.weights)