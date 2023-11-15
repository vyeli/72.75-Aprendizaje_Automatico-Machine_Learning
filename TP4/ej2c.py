import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from KMeans import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from distance_matrix import calculate_distance_matrix

def cluster_precision(cluster_df, n_clusters):
    classes = [{
                'Action': 0,
                'Comedy': 0,            
                'Drama': 0
            } for _ in range(n_clusters)]
    for movie in cluster_df.values:
        cluster = int(movie[0])
        classes[cluster][movie[1]] += 1
    print(classes)

def movies_in_cluster(clusters, cluster_idx):
    return clusters[clusters[0] == cluster_idx]


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


movie_df = pd.read_csv('data/filtered_movie_data.csv', sep=';').dropna()
genres = movie_df['genres'].values
movies = movie_df['original_title'].values
budget = movie_df['budget'].values
popularity = movie_df['popularity'].values
revenue = movie_df['revenue'].values
runtime = movie_df['runtime'].values
vote_average = movie_df['vote_average'].values
vote_count = movie_df['vote_count'].values
movie_df = movie_df.select_dtypes(include=[np.number])[['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']]
print(movie_df.head())
movie_df = pd.DataFrame(StandardScaler().fit_transform(movie_df))
print(movie_df.head())

# KMeans
# KMeans.plot_elbow_method(movie_df.values, 1, 10)

# Hierarchical Clustering
distance_matrix = calculate_distance_matrix(movie_df.values)

linked_matrix = linkage(distance_matrix, method="average")

fancy_dendrogram(linked_matrix, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True, annotate_above=10, no_plot=False)

clusters = pd.DataFrame(fcluster(linked_matrix, 87.1, criterion='distance'))
clusters['genres'] = genres
clusters['movies'] = movies
clusters['budget'] = budget
clusters['popularity'] = popularity
clusters['revenue'] = revenue
clusters['runtime'] = runtime
clusters['vote_average'] = vote_average
clusters['vote_count'] = vote_count
clusters = clusters.sort_values(by=[0], ascending=True)

cluster_precision(clusters, np.max(clusters[0])+1)

for cluster in [1, 2, 3, 6, 8, 9, 10, 11]:
    print(movies_in_cluster(clusters, cluster))
    print()

# n_clusters = 6
# k = KMeans(K=n_clusters, max_iters=300)
# y_pred = k.predict(movie_df.values)

# y_pred = pd.DataFrame(y_pred, columns=['cluster'])
# y_pred['genres'] = genres

# y_pred = y_pred.sort_values(by=['cluster'], ascending=True)
# y_pred.to_csv('data/movie_clusters.csv', index=False)
# cluster_precision(y_pred, n_clusters)
