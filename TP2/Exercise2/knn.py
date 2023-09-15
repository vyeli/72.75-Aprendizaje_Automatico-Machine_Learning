
class KNN():
    def _init_(self, k, distance, weighted, weighted_distance, voting, voting_distance):
        self.k = k
        self.distance = distance
        self.weighted = weighted
        self.weighted_distance = weighted_distance
        self.voting = voting
        self.voting_distance = voting_distance


    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._predict(x))
        return y_pred