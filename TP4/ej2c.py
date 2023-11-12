import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from KMeans import KMeans

def cluster_precision(cluster_df, n_clusters):
    classes = [{
                'Action': 0,
                'Comedy': 0,            
                'Drama': 0
            } for i in range(n_clusters)]
    for movie in cluster_df.values:
        cluster = int(movie[0])
        classes[cluster][movie[1]] += 1
    print(classes)

# KMeans
movie_df = pd.read_csv('data/filtered_movie_data.csv', sep=';').dropna()
genres = movie_df['genres'].values
movie_df = movie_df.select_dtypes(include=[np.number])
print(movie_df.head())
# movie_df = movie_df[['revenue', 'popularity']]

n_clusters = 10
k = KMeans(K=n_clusters, max_iters=150)
y_pred = k.predict(movie_df.values)

y_pred = pd.DataFrame(y_pred, columns=['cluster'])
y_pred['genres'] = genres

y_pred = y_pred.sort_values(by=['cluster'], ascending=True)
y_pred.to_csv('data/movie_clusters.csv', index=False)
print(cluster_precision(y_pred, n_clusters))
# k.plot()