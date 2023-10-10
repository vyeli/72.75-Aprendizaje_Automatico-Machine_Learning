from sklearn import datasets

def generate_linearly_separable_data(n_samples):
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=2, cluster_std=1.5, random_state=42)
    return X, y

def generate_not_so_linearly_separable_data(n_samples):
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=2, centers=2, cluster_std=4.6, random_state=42)
    return X, y