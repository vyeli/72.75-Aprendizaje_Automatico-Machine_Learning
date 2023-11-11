from scipy.spatial.distance import pdist, squareform


def calculate_distance_matrix(data):
    # Assuming 'data' is your input data
    distance_matrix = pdist(data, metric='euclidean')
    distance_matrix = squareform(distance_matrix)  # Convert to a square matrix
    return distance_matrix
