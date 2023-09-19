import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from ID3 import DecisionTree
from RandomForest import RandomForest
from metrics import MetricsCalculator
from sklearn.model_selection import train_test_split

# Create the output folder if it doesn't exist
if not os.path.exists("Output/ex1"):
    os.makedirs("Output/ex1")

# Read the CSV file into a DataFrame and clean the data (remove the rows with missing values)
df = pd.read_csv("Data/german_credit.csv", delimiter=",")
# split the data into features and target variable where the target column is Creditability
X = df.drop("Creditability", axis=1).to_numpy()
y = df["Creditability"].to_numpy()

heights = [2+i for i in range(20)]

train_accuracies_mean = []
train_accuracies_std = []

test_accuracies_mean = []
test_accuracies_std = []

nodes_mean = []
nodes_std = []

for height in heights:
    train_accuracies = []
    test_accuracies = []
    nodes = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        #decision tree
        decision_tree = DecisionTree(max_depth=height)
        decision_tree.fit(X_train, y_train)
        predictions = decision_tree.predict(X_test)
        nodes.append(decision_tree.get_Total_nodes())

        metrics = MetricsCalculator()
        train_accuracies.append(metrics.accuracy(y_train, decision_tree.predict(X_train)))
        test_accuracies.append(metrics.accuracy(y_test, predictions))
    
    train_accuracies_mean.append(np.mean(train_accuracies))
    train_accuracies_std.append(np.std(train_accuracies))

    test_accuracies_mean.append(np.mean(test_accuracies))
    test_accuracies_std.append(np.std(test_accuracies))

    nodes_mean.append(np.mean(nodes))
    nodes_std.append(np.std(nodes))

fig, ax = plt.subplots()

sc = ax.scatter(nodes_mean, heights, c=train_accuracies_mean, cmap=plt.cm.plasma, vmin=np.min(train_accuracies_mean), vmax=np.max(train_accuracies_mean))
plt.colorbar(sc, label="Train accuracy")

ax.errorbar(nodes_mean, heights, xerr=nodes_std, fmt='none', color="gray", alpha=0.5)

plt.ylabel("Tree height")
plt.xlabel("Nodes")
plt.savefig("Output/ex1/precision_vs_train_nodes.png")
plt.show()
plt.clf()

fig, ax = plt.subplots()
sc = ax.scatter(nodes_mean, heights, c=test_accuracies_mean, cmap=plt.cm.plasma, vmin=np.min(test_accuracies_mean), vmax=np.max(test_accuracies_mean))
plt.colorbar(sc, label="Test accuracy")

ax.errorbar(nodes_mean, heights, xerr=nodes_std, fmt='none', color="gray", alpha=0.5)

plt.ylabel("Tree height")
plt.xlabel("Nodes")
plt.savefig("Output/ex1/precision_vs_test_nodes.png")
plt.show()