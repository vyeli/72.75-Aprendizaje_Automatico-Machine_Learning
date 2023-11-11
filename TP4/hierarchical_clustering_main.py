import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.cluster.hierarchy import linkage, dendrogram

from distance_matrix import calculate_distance_matrix

if not os.path.exists("plots"):
    os.mkdir("plots")

# Importing the dataset
df = pd.read_csv('data/movie_data.csv', sep=';')

numeric_df = df.select_dtypes(include=[np.number])

# Remove null values
numeric_df = numeric_df.dropna()

# Standardize data to give it to the clustering algorithm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_df = scaler.fit_transform(numeric_df)  # standardize all features to have mean=0 and sd=1

# Calculate distance matrix
distance_matrix = calculate_distance_matrix(numeric_df)

# Perform hierarchical/agglomerative clustering
linked_matrix = linkage(distance_matrix, method="average")

dendrogram(linked_matrix, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True, show_leaf_counts=True)
plt.ylabel("Distance")
plt.title("Dendrogram")
plt.savefig("plots/dendrogram.png")

