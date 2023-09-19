import numpy as np
from utils import Distance
from sortedcontainers import SortedList

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KNN():
    def __init__(self, k, weighted: bool = False):
        self.k = k
        self.weighted = weighted

    def predict(self, X_test, X_train, y_train):
        if self.weighted:
            return self.weighted_predict(X_test, X_train, y_train)
        else:
            return self.simple_predict(X_test, X_train, y_train)
        
    def simple_predict(self, X_test, X_train, y_train):
        y_pred = []
        for i in range(len(X_test)):
            distances = SortedList([])
            for j in range(len(X_train)):
                distances.add(Distance(euclidean_distance(X_test[i], X_train[j]), j))
            k_nearest = [distances[i].index for i in range(self.k)]

            y_pred.append(np.bincount(y_train[k_nearest]).argmax())
        return np.array(y_pred)

    def weighted_predict(self, X_test, X_train, y_train):
        y_pred = []
        for i in range(len(X_test)):
            distances = SortedList([])
            for j in range(len(X_train)):
                distances.add(Distance(euclidean_distance(X_test[i], X_train[j]), j))
            
            k_nearest = []
            k_distances = []
            for w in range(self.k):
                k_nearest.append(distances[w].index)
                k_distances.append(distances[w].distance)
            
            # avoid division by zero
            if k_distances[0] == 0:
                y_pred.append(y_train[k_nearest[0]]) # Choose the class of the nearest neighbor
            else:
                weights = [1 / distance for distance in k_distances]
                y_pred.append(np.bincount(y_train[k_nearest], weights).argmax())
            
        return np.array(y_pred)
