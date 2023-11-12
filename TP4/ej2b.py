import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KMeans import KMeans



def plot_elbow_method(data):
    sse = []
    for k in range (3, 20):
        kmeans = KMeans(K=k, max_iters=300)
        kmeans.predict(data.values)
        sse.append(kmeans.inertia())
    
    plt.plot(range(3, 20), sse)
    plt.xticks(range(3, 20))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
