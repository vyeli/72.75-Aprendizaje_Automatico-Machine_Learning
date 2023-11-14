import numpy as np
import matplotlib.pyplot as plt

class Kohonen:
    def __init__(self, map_size, n_features, learning_rate=0.1, Ri=1.0, n_iterations=100, random_seed=None):
        self.map_size = map_size
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.Ri = Ri
        self.n_iterations = n_iterations
        self.current_iteration = 0
        self.random_seed = random_seed
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.weights = np.random.rand(map_size[0], map_size[1], n_features)
        
    def _get_bmu(self, sample):
        distances = np.sum((self.weights - sample)**2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def calculate_R(self):
        return (self.n_iterations-self.current_iteration) * self.Ri / self.n_iterations
        
    def _get_neighborhood(self, bmu_idx):
        x, y = np.indices(self.map_size)
        distance = np.sqrt((x - bmu_idx[0])**2 + (y - bmu_idx[1])**2)
        return np.exp(-2*distance / self.calculate_R())
        
    def _update_weights(self, sample, bmu_idx, lr):
        neighborhood = self._get_neighborhood(bmu_idx)
        delta = lr * neighborhood[:, :, np.newaxis] * (sample - self.weights)
        self.weights += delta
        
    def fit(self, X):        
        for i in range(self.n_iterations):
            lr = self.learning_rate * np.exp(-i / self.n_iterations)
            for sample in X:
                bmu_idx = self._get_bmu(sample)
                lr *= np.random.uniform(0.01, 2)
                self._update_weights(sample, bmu_idx, lr)
            self.current_iteration += 1
                
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, sample in enumerate(X):
            bmu_idx = self._get_bmu(sample)
            y_pred[i] = bmu_idx[0] * self.map_size[1] + bmu_idx[1]
        return y_pred
    
    def plot_u_matrix(self): #a chequear
        u_matrix = np.zeros((self.map_size[0]*2-1, self.map_size[1]*2-1))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if i == 0 and j == 0:
                    u_matrix[0, 0] = np.mean(np.sqrt(np.sum((self.weights[i, j] - self.weights[i, j+1])**2)))
                elif i == 0 and j == self.map_size[1]-1:
                    u_matrix[0, -1] = np.mean(np.sqrt(np.sum((self.weights[i, j] - self.weights[i, j-1])**2)))
                elif i == self.map_size[0]-1 and j == 0:
                    u_matrix[-1, 0] = np.mean(np.sqrt(np.sum((self.weights[i, j] - self.weights[i-1, j])**2)))
                elif i == self.map_size[0]-1 and j == self.map_size[1]-1:
                    u_matrix[-1, -1] = np.mean(np.sqrt(np.sum((self.weights[i, j] - self.weights[i-1, j])**2)))
                elif i == 0:
                    u_matrix[0, j*2] = np.mean(np.sqrt(np.sum((self.weights[i, j] - self.weights[i, j-1:j+2])**2)))
                elif i == self.map_size[0]-1:
                    u_matrix[-1, j*2] = np.mean(np.sqrt(np.sum((self.weights[i, j] - self.weights[i-1, j-1:j+2])**2)))
                elif j == 0:
                    u_matrix[i*2, 0] = np.mean(np.sqrt(np.sum((self.weights[i, j] - self.weights[i-1:i+2, j])**2)))
                elif j == self.map_size[1]-1:
                    u_matrix[i*2, -1] = np.mean(np.sqrt(np.sum((self.weights[i, j] - self.weights[i-1:i+2, j-1:j+1])**2)))
                else:
                    u_matrix[i*2, j*2] = np.mean(np.sqrt(np.sum((self.weights[i, j] - self.weights[i-1:i+2, j-1:j+2])**2)))
        
        plt.imshow(u_matrix, cmap='YlOrRd')
        plt.colorbar()
        plt.show()