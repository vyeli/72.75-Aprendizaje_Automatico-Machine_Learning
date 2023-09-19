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

split_percentages = [0.1 * i for i in range(1, 10)]

train_non_weighted_accuracies_mean = []
train_non_weighted_accuracies_err = []
train_weighted_accuracies_mean = []
train_weighted_accuracies_err = []

non_weighted_accuracies_mean = []
non_weighted_accuracies_err = []
weighted_accuracies_mean = []
weighted_accuracies_err = []

weighted_accuracy = []
non_weighted_accuracy = []

for percentage in split_percentages:
    train_non_weighted_accuracies = []
    non_weighted_accuracies = []
    
    train_weighted_accuracies = []
    weighted_accuracies = []
    for i in range(20):
        # Split the data into train and test sets
        train_set, test_set = train_test_split(normalized_df, test_size=percentage)

        # extract the labels from the train and test sets
        y_train = train_set.pop("Star Rating")
        y_test = test_set.pop("Star Rating")

        # Convert the DataFrames to numpy arrays
        train_set = train_set.values
        test_set = test_set.values

        y_train = y_train.values
        y_test = y_test.values

        no_weight_knn = KNN(k=5, weighted=False)

        y_pred = no_weight_knn.predict(test_set, train_set, y_train)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        non_weighted_accuracies.append(accuracy)

        if percentage == 0.4:
            non_weighted_accuracy.append(accuracy)

        y_pred_train = no_weight_knn.predict(train_set, train_set, y_train)
        accuracy = np.sum(y_pred_train == y_train) / len(y_train)
        train_non_weighted_accuracies.append(accuracy)

        weighted_knn = KNN(k=5, weighted=True)

        y_pred = weighted_knn.predict(test_set, train_set, y_train)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        weighted_accuracies.append(accuracy)

        if percentage == 0.4:
            weighted_accuracy.append(accuracy)

        y_pred_train = weighted_knn.predict(train_set, train_set, y_train)
        accuracy = np.sum(y_pred_train == y_train) / len(y_train)
        train_weighted_accuracies.append(accuracy)
    
    non_weighted_accuracies = np.array(non_weighted_accuracies)
    weighted_accuracies = np.array(weighted_accuracies)
    train_non_weighted_accuracies = np.array(train_non_weighted_accuracies)
    train_weighted_accuracies = np.array(train_weighted_accuracies)

    non_weighted_accuracies_mean.append(non_weighted_accuracies.mean())
    non_weighted_accuracies_err.append(non_weighted_accuracies.std())
    weighted_accuracies_mean.append(weighted_accuracies.mean())
    weighted_accuracies_err.append(weighted_accuracies.std())

    train_non_weighted_accuracies_mean.append(train_non_weighted_accuracies.mean())
    train_non_weighted_accuracies_err.append(train_non_weighted_accuracies.std())
    train_weighted_accuracies_mean.append(train_weighted_accuracies.mean())
    train_weighted_accuracies_err.append(train_weighted_accuracies.std())

plt.errorbar(split_percentages, non_weighted_accuracies_mean, yerr=non_weighted_accuracies_err, label="Test", marker='o', linestyle="dashed")
plt.errorbar(split_percentages, train_non_weighted_accuracies_mean, yerr=train_non_weighted_accuracies_err, label="Train", marker='o', linestyle="dashed")
plt.ylabel("Accuracy")
plt.xlabel("Test set %")
plt.title("Non weighted split")
plt.legend()
plt.savefig("Output/ex2/non_weighted_test_train_split.png")
plt.show()

plt.errorbar(split_percentages, weighted_accuracies_mean, yerr=weighted_accuracies_err, label="Test", marker='o', linestyle="dashed")
plt.errorbar(split_percentages, train_non_weighted_accuracies_mean, yerr=train_weighted_accuracies_err, label="Train", marker='o', linestyle="dashed")
plt.ylabel("Accuracy")
plt.xlabel("Test set %")
plt.title("Weighted split")
plt.legend()
plt.savefig("Output/ex2/weighted_test_train_split.png")
plt.show()

print("Non weighted accuracy")
print(f"\tvalue:{np.mean(non_weighted_accuracy)}")
print(f"\tstd:{np.std(non_weighted_accuracy)}")
print("Weighted accuracy")
print(f"\tvalue:{np.mean(weighted_accuracy)}")
print(f"\tstd:{np.std(weighted_accuracy)}")