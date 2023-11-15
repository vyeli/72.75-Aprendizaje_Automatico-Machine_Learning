import numpy as np

class Kohonen:
    def __init__(self, n_rows, n_cols, n_features, sigma=1.0, learning_rate=0.5, seed=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_features = n_features
        self.sigma = sigma
        self.learning_rate = learning_rate
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.uniform(low=-1,high=1,size=(n_rows, n_cols, n_features))

    def train(self, data, n_iterations):
        for i in range(n_iterations):
            # select a random data point
            x = data[np.random.randint(data.shape[0]), :]

            # find the best matching unit (BMU)
            bmu_row, bmu_col = self._find_bmu(x)

            # update the weights of the BMU and its neighbors
            self._update_weights(x, bmu_row, bmu_col, i, n_iterations)

    def _find_bmu(self, x):
        # compute the distance between x and each weight vector
        distances = np.sum((self.weights - x) ** 2, axis=2)

        # find the row and column indices of the BMU
        bmu_row, bmu_col = np.unravel_index(np.argmin(distances), distances.shape)

        return bmu_row, bmu_col

    def _update_weights(self, x, bmu_row, bmu_col, iteration, n_iterations):
        # compute the neighborhood function
        sigma_t = self.sigma * np.exp(-iteration / n_iterations)
        distance = np.sqrt((np.arange(self.n_rows) - bmu_row) ** 2 + (np.arange(self.n_cols)[:, np.newaxis] - bmu_col) ** 2)
        neighborhood = np.exp(-distance ** 2 / (2 * sigma_t ** 2))

        # update the weights of the BMU and its neighbors
        learning_rate_t = self.learning_rate * np.exp(-iteration / n_iterations)
        delta = learning_rate_t * neighborhood[:, :, np.newaxis] * (x - self.weights)
        self.weights += delta

    def distance_map(self):
        # compute the distance map
        distance_map = np.zeros((self.n_rows, self.n_cols))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                w = self.weights[i, j, :]
                distances = np.sum((self.weights - w) ** 2, axis=2)
                distance_map[i, j] = np.sqrt(np.mean(distances))

        return distance_map
    
    def distance_map_feature(self, feature_index):
        # compute the distance map for the specified feature index
        distance_map = np.zeros((self.n_rows, self.n_cols))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                w = self.weights[i, j, feature_index]
                distances = np.sum((self.weights[:, :, feature_index] - w) ** 2, axis=0)
                distance_map[i, j] = np.sqrt(np.mean(distances))

        return distance_map