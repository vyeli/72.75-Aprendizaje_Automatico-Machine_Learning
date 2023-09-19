import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# B split the data into test and train, and normalize the data
from sklearn.model_selection import train_test_split
from utils import confusion_matrix
from knn import KNN

# A 
# Read the CSV file into a DataFrame and clean the data (remove the rows with missing values)
df = pd.read_csv("Data/reviews_sentiment.csv", delimiter=";")

# Remove the rows with missing values
df = df.dropna()
# Remove the columns of Review Title, Review Text and textSentiment (we don't need use them for this exercise)
df = df.drop(columns=["Review Title", "Review Text", "textSentiment"])
# Replace the values in the "titleSentiment" column with 0 or 1
df["titleSentiment"] = df["titleSentiment"].map({"negative": 0, "positive": 1})

# normalize the data
def standar_normalize(data):
    return (data - data.mean()) / data.std()

def feature_scaling(data, a=0, b=1):
    return (data - data.min()) * (b - a) / (data.max() - data.min()) + a

columns_to_normalize = ["wordcount", "titleSentiment", "sentimentValue"]

normalized_df = df.copy()

normalized_df[columns_to_normalize] = normalized_df[columns_to_normalize].apply(lambda x: standar_normalize(x))

k_values = [1 + 2*i for i in range(13)]

non_weighted_k_accuracies_mean = []
weighted_k_accuracies_mean = []

non_weighted_k_accuracies_err = []
weighted_k_accuracies_err = []

for k in k_values:
    non_weighted_k_accuracies = []
    weighted_k_accuracies = []
    for i in range(20):
        # Split the data into train and test sets
        train_set, test_set = train_test_split(normalized_df, test_size=0.4)

        # extract the labels from the train and test sets
        y_train = train_set.pop("Star Rating")
        y_test = test_set.pop("Star Rating")

        # Convert the DataFrames to numpy arrays
        train_set = train_set.values
        test_set = test_set.values

        y_train = y_train.values
        y_test = y_test.values

        no_weight_knn = KNN(k=k, weighted=False)

        y_pred = no_weight_knn.predict(test_set, train_set, y_train)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        non_weighted_k_accuracies.append(accuracy)

        weighted_knn = KNN(k=k, weighted=True)

        y_pred = weighted_knn.predict(test_set, train_set, y_train)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        weighted_k_accuracies.append(accuracy)
    
    non_weighted_k_accuracies = np.array(non_weighted_k_accuracies)
    weighted_k_accuracies = np.array(weighted_k_accuracies)

    non_weighted_k_accuracies_mean.append(non_weighted_k_accuracies.mean())
    non_weighted_k_accuracies_err.append(non_weighted_k_accuracies.std())
    weighted_k_accuracies_mean.append(weighted_k_accuracies.mean())
    weighted_k_accuracies_err.append(weighted_k_accuracies.std())

plt.errorbar(k_values, non_weighted_k_accuracies_mean, yerr=non_weighted_k_accuracies_err, label="Non weighted", marker='o', linestyle="dashed")
plt.errorbar(k_values, weighted_k_accuracies_mean, yerr=weighted_k_accuracies_err, label="Weighted", marker='o', linestyle="dashed")
plt.ylabel("Accuracy")
plt.xlabel("K")
plt.xticks(k_values)
plt.title("Accuracy vs K")
plt.legend()
plt.savefig("Output/ex2/k_values.png")
plt.show()
