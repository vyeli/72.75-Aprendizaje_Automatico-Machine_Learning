import numpy as np

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
            distances = []
            for j in range(len(X_train)):
                distances.append(euclidean_distance(X_test[i], X_train[j]))
            distances = np.array(distances)
            sorted_indexes = np.argsort(distances) 
            k_nearest = sorted_indexes[:self.k]
            y_pred.append(np.bincount(y_train[k_nearest]).argmax())
        return np.array(y_pred)

    def weighted_predict(self, X_test, X_train, y_train):
        y_pred = []
        for i in range(len(X_test)):
            distances = []
            for j in range(len(X_train)):
                distances.append(euclidean_distance(X_test[i], X_train[j]))
            distances = np.array(distances)
            sorted_indexes = np.argsort(distances) 
            k_nearest = sorted_indexes[:self.k]
            k_distances = distances[k_nearest]
            
            # avoid division by zero
            if k_distances[0] == 0:
                y_pred.append(y_train[k_nearest[0]]) # Choose the class of the nearest neighbor
            else:
                weights = 1 / k_distances
                y_pred.append(np.bincount(y_train[k_nearest], weights).argmax())
            
        return np.array(y_pred)
