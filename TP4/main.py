import pandas as pd
import matplotlib.pyplot as plt
from Kohonen import Kohonen

# cargar el dataset en un DataFrame
df = pd.read_csv('data/filtered_movie_data.csv', sep=";").dropna()
#normalized_df = df[df["genres"] .isin(["Action", "Comedy"])]
normalized_df = df
normalized_df = normalized_df.drop(['genres', 'original_title', 'overview', 'imdb_id', 'release_date'], axis=1)
normalized_df = (normalized_df-normalized_df.min())/(normalized_df.max()-normalized_df.min())

# convertir el DataFrame a un array NumPy
data = normalized_df.values

# create a SOM
som = Kohonen(15, 15, data.shape[1], seed= 35)

# train the SOM on the dataset for 1000 iterations
som.train(data, 100000)

# compute the distance map of the SOM
distance_map = som.distance_map()

# visualize the distance map
plt.imshow(distance_map, cmap='YlOrRd')
plt.colorbar()
plt.title("Action + Comedy")
plt.show()

