import numpy as np

class MetricsCalculator:

    def __init__(self):
        pass

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)