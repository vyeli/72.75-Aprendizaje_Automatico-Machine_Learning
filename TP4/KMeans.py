import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, K= 5, max_iters=100, plot_steps=False) -> None:
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
    
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        #initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        #optimize clusters
        for _ in range(self.max_iters):
            #update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            #update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot()

            #check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        #return cluster labels
        return self._get_cluster_labels(self.clusters)
    
    def _is_converged(self, centroids_old, centroids):
        distances = [self._euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
    
    def _closest_centroid(self, sample, centroids):
        distances = [self._euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)
        # return np.sqrt(np.sum((x1 - x2)**2)) #to be modified
    
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def inertia(self):
        inertia = 0
        for cluster_idx, cluster in enumerate(self.clusters):
            centroid = self.centroids[cluster_idx]
            inertia += np.sum(np.linalg.norm(self.X[cluster] - centroid) ** 2) / len(cluster)
        return inertia
    
    def plot_elbow_method(data, min=3, max=20, output_file='output.png'):
        sse = []
        for k in range (min, max):
            kmeans = KMeans(K=k, max_iters=300)
            kmeans.predict(data)
            sse.append(kmeans.inertia())
        
        plt.plot(range(min, max), sse)
        plt.xticks(range(min, max))
        plt.xlabel("Cantidad de clusters")
        plt.ylabel("WCSS")
        plt.savefig(output_file)
        plt.show()
