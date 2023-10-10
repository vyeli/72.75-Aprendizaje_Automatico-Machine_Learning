import numpy as np

def _unit_step_func(x):
    return np.where(x>=0, 1, 0)

class SimplePerceptron:
    def __init__(self, learning_rate = 0.01, epochs=1000, accepctable_error=0.001):
        self.lr = learning_rate
        self.epochs = epochs
        self.activation_func = _unit_step_func
        self.accepctable_error = accepctable_error
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        errors = []
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        # learning
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
            
            actual_error = self.error(X, y)
            errors.append(actual_error)
            if actual_error <= self.accepctable_error:
                print("Converged at epoch: ", _)
                return _ + 1, actual_error, errors
        print("Did not converge")
        return self.epochs, actual_error, errors     
    
    def predict(self, X):
        # hyperplane equation
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
    
    def error(self, X, y):
        y_ = np.where(y > 0, 1, 0)
        y_predicted = self.predict(X)
        return np.mean(np.abs(y_ - y_predicted))

        