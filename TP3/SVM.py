import numpy as np

class SVM_classifier:

    def __init__(self, learning_rate, iterations, lambda_parameter) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, y):
        self.rows, self.cols = X.shape
        self.weights = np.zeros(self.cols)
        self.bias = 0

        #implementing gradient descent
        for i in range(self.iterations):
            self.update_weights(X, y)

    def update_weights(self, X, y):
        y_label = np.where(y <= 0, -1, 1)

        for i, x in enumerate(X):
            condition = y_label[i] * (np.dot(x, self.weights) - self.bias) >= 1
            if (condition):
                dw = 2 * self.lambda_parameter * self.weights
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.weights - np.dot(x, y_label[i])
                db = y_label[i]
        
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        output = np.dot(X, self.weights) - self.bias
        predicted_labels = np.sign(output)
        #revert np.where() from fit method
        predicted_labels = np.where(predicted_labels <= 0, 0, 1)
        return predicted_labels